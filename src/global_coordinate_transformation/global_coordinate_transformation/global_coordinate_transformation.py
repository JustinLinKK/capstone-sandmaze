import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from sensor_msgs.msg import PointField
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import numpy as np
import struct
import os

class GlobalCoordinateTransformation(Node):
    def __init__(self):
        super().__init__('GlobalCoordinateTransformation')
        # Lidar publishes as best effort , the scan topic so i had to add this to make sure data can be accessed.
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT, 
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',  # odom topic, switch to output of ekf or camera once. Currently set to odom as second Lidar laserscan matcher was configured to publish odometry on /odom
            self.odom_callback,
            qos_profile
        )
        self.lidar_sub = self.create_subscription(
            LaserScan,
            '/scan',   # lidar topic is /scan but as i was testing using two lidars i had the /scan2 associated with vertically placed lidar.  
            self.lidar_callback,
            qos_profile
        )

        # Publisher
        self.pointcloud_pub = self.create_publisher(PointCloud2, '/GlobalCoordinateTransformation', qos_profile)

        # for pose diff calculations 
        self.current_pose = None
        self.last_saved_pose = None

        # Movement thresholds to avoid redundant data
        self.position_threshold = 0.1 # meters smaller threshold better but more noisy
        self.orientation_threshold = 3  # degrees keep above than 1 for less noisy data
        
    def odom_callback(self, msg):
        self.current_pose = {
            'x': msg.pose.pose.position.x,
            'y': msg.pose.pose.position.y,
            'z': msg.pose.pose.position.z,
            'qx': msg.pose.pose.orientation.x,
            'qy': msg.pose.pose.orientation.y,
            'qz': msg.pose.pose.orientation.z,
            'qw': msg.pose.pose.orientation.w,
        }

    def lidar_callback(self, msg):
        if self.current_pose is None:
            self.get_logger().warning("No odometry data received yet.") #no odometry indicates non-functioning components 
            return
        if not self.move_detect():
            #self.get_logger().info("not moving , testing")
            return  # Skip saving if movement is too small
        self.get_logger().info("saving data movement changed")
        points = self.scan_to_frame(msg)
        transformed_points = self.transform_points(points, self.current_pose)
        self.save_data(transformed_points)
        self.last_saved_pose = self.current_pose.copy()
        pointcloud_msg = self.create_pointcloud2(transformed_points)
        self.pointcloud_pub.publish(pointcloud_msg)

    
    def scan_to_frame(self, scan_msg): 
        points = []
        angle = scan_msg.angle_min
        # the Lidar driver can be set to publish nan/inf as zeroes but with pose translation then points will appear at current postion
        # so i set it the Lidar driver nan/inf to zero as false, but filtered them out in here
        # in the IMU implmentation you could just change the nan/inf =  0 as true as imu + lidar handling is just rotmatrix*point with no translation
        for r in scan_msg.ranges:
            if not np.isfinite(r) or r <= 0 or r > scan_msg.range_max:
                angle += scan_msg.angle_increment
                continue  
            x=0.0 # local frame 
            y = r * np.cos(angle)  # LiDAR Y-axis
            z = r * np.sin(angle)  # LiDAR Z-axis
            points.append((x, y, z))  
            angle += scan_msg.angle_increment
        return np.array(points)
        
    def move_detect(self):
        if self.last_saved_pose is None:
            return True  #first scan
        last_pos = np.array([self.last_saved_pose['x'], self.last_saved_pose['y'], self.last_saved_pose['z']])
        curr_pos = np.array([self.current_pose['x'], self.current_pose['y'], self.current_pose['z']])
        position_change = np.linalg.norm(curr_pos - last_pos)
        last_quat = np.array([self.last_saved_pose['qx'], self.last_saved_pose['qy'], self.last_saved_pose['qz'], self.last_saved_pose['qw']])
        curr_quat = np.array([self.current_pose['qx'], self.current_pose['qy'], self.current_pose['qz'], self.current_pose['qw']])
        quat_diff = self.quat_multiply(curr_quat, self.quat_inv(last_quat))
        orientation_change = 2 * np.arccos(np.clip(quat_diff[3], -1.0, 1.0)) * (180.0 / np.pi)  
        return position_change > self.position_threshold or orientation_change > self.orientation_threshold
        
    def rotation_matrix(self, quaternion): 
        x=quaternion[0]
        y=quaternion[1]
        z =quaternion[2]
        w = quaternion[3]
        return np.array([
            [1 - 2 * (y**2+ z**2), 2* (x*y - z*w), 2 * (x*z + y*w)],
            [2 * (x * y + z * w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
            [2*(x * z - y * w), 2*(y*z + x*w), 1 - 2 * (x**2 + y**2)]
        ])
    def quat_multiply(self, q1, q2):
        x1=q1[0]
        y1 =q1[1]
        z1 =q1[2]
        w1 = q1[3]
        x2=q2[0]
        y2 =q2[1]
        z2=q2[2]
        w2 =q2[3]
        w = w1*w2 - x1 *x2 - y1 * y2 - z1 * z2
        x = w1*x2 +x1* w2 +y1 * z2 - z1 * y2
        y = w1*y2 - x1 * z2+ y1 *w2 + z1 * x2
        z = w1*z2 +x1 * y2 - y1* x2 + z1 * w2
        return np.array([x, y, z, w])
        
    def quat_inv(self, quaternion):
        return np.array([-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3])

    def save_data(self, points):
        filename = 'transformed_points.xyz'
        with open(filename, 'a') as file:
            for x, y, z in points:
                file.write(f"{x} {y} {z}\n")
    
    # Transforms the points from local frame to global frame 
    def transform_points(self, points, pose):
        position = np.array([pose['x'], pose['y'], pose['z']])
        orientation = [pose['qx'], pose['qy'], pose['qz'], pose['qw']]
        rotation_matrix = self.rotation_matrix(orientation)
        transformed_points = [np.dot(rotation_matrix, point) + position for point in points]
        return transformed_points # Tested with lcoalization using a second lidar utilizing position tracking , points correclty transformed to new position
    
    # the pointcloud 2 message is not necessary and only used for debug to visualize point cloud in real time in RVIZ
    # in RVIZ change to point cloud , change topic reliability to best effort
    # change frame to odom topic your using and points should appear 
    # TODO RVIZ config that launches with the launch file so person could immediately have everything setup [X] debug only

    # the pointcloud 2 message is not necessary and only used for debug to visualize point cloud in real time in RVIZ, comment out for faster perfromance
    def create_pointcloud2(self, points): 
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'odom' 
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        point_data = []
        for x, y, z in points:
            point_data.extend(struct.pack('fff', x, y, z))
        pointcloud_msg = PointCloud2()
        pointcloud_msg.header = header  
        pointcloud_msg.height = 1
        pointcloud_msg.width = len(points)
        pointcloud_msg.fields = fields
        pointcloud_msg.is_bigendian = False
        pointcloud_msg.point_step = 12  
        pointcloud_msg.row_step = pointcloud_msg.point_step * len(points)
        pointcloud_msg.data = bytearray(point_data)
        pointcloud_msg.is_dense = True
        return pointcloud_msg

def main(args=None):
    rclpy.init(args=args)
    node = GlobalCoordinateTransformation()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

