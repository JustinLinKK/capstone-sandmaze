import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2, Imu
from std_msgs.msg import Header
from sensor_msgs.msg import PointField
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
import numpy as np
import struct

class VerticalLidarMapper(Node):
    def __init__(self):
        super().__init__('vertical_lidar_mapper')

        qos_profile = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT,
            history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.imu_received = False
        self.lidar_received = False

        self.imu_sub = self.create_subscription(
            Imu,
            '/bno055/imu',  
            self.imu_callback,
            qos_profile
        )
        self.lidar_sub = self.create_subscription(
            LaserScan,
            '/scan',  
            self.lidar_callback,
            qos_profile
        )

        self.pointcloud_pub = self.create_publisher(PointCloud2, '/vertical_lidar_pointcloud', qos_profile)

        # TF2 Buffer and Listener (TO do  to use for synchronization but have not done the implementation  yet)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.tf_broadcaster = TransformBroadcaster(self)
        self.current_orientation = None
        self.initial_orientation = None  # Store the first IMU orientation
        self.orientation_threshold = 3  # Threshold for orientation change to trigger saving
        self.last_saved_orientation = None

        # Timer to check for missing messages
        self.create_timer(2.0, self.check_received_messages)

    def check_received_messages(self):
        if not self.imu_received:
            self.get_logger().warning("No IMU messages received")
        if not self.lidar_received:
            self.get_logger().warning("No LiDAR scans received")

    def imu_callback(self, msg):
        self.imu_received = True
        self.current_orientation = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])

        if self.initial_orientation is None:
            self.initial_orientation = self.current_orientation
            self.get_logger().info("Stored initial IMU orientation.")
            
        rel_orientation = self.relative_orientation(self.current_orientation, self.initial_orientation)
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = 'map'
        transform.child_frame_id = 'Frame3d'
        transform.transform.rotation.x = rel_orientation[0]
        transform.transform.rotation.y = rel_orientation[1]
        transform.transform.rotation.z = rel_orientation[2]
        transform.transform.rotation.w = rel_orientation[3]
        self.tf_broadcaster.sendTransform(transform)

    def lidar_callback(self, msg):
        self.lidar_received = True
        if self.current_orientation is None:
            self.get_logger().warning("No IMU data received yet.")
            return
        points = self.scan_to_frame(msg)
        transformed_points = self.transform_points(points, self.current_orientation)
        if self.orientation_change():
            self.get_logger().info("orientation change detected ,saving data")
            self.save_data(transformed_points)
            self.last_saved_orientation = self.current_orientation.copy()
            pointcloud_msg = self.create_pointcloud2(transformed_points)
            self.pointcloud_pub.publish(pointcloud_msg)
    
    
    def orientation_change(self):
        if self.last_saved_orientation is None:
            return True # for first scan to be saved
        q1_inv = self.quat_inv(self.last_saved_orientation)
        q_diff = self.quat_multiply(q1_inv, self.current_orientation)
        diff_rad = 2 * np.arccos(np.clip(q_diff[3], -1.0, 1.0))
        diff_deg = np.degrees(diff_rad)
        return diff_deg > self.orientation_threshold

    def scan_to_frame(self, scan_msg):
        points = []
        angle = scan_msg.angle_min
        for r in scan_msg.ranges:
            if scan_msg.range_min < r < scan_msg.range_max:
                x=0.0
                y = r * np.cos(angle)
                z = r * np.sin(angle)
                points.append((x, y, z))
            angle += scan_msg.angle_increment
        return np.array(points)

        
    def quat_inv(self, quaternion):
        
        return np.array([-quaternion[0], -quaternion[1], quaternion[2], quaternion[3]])
    
    def quat_multiply(self, q1, q2):
        x1=q1[0]
        y1=q1[1]
        z1=q1[2]
        w1 = q1[3]
        
        x2=q2[0]
        y2=q2[1]
        z2=q2[2]
        w2 = q2[3]
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return np.array([x, y, z, w])
    # relative orientation ( need to check if needed given imu is calibrated after power on)
    def relative_orientation(self, current, initial): 
        initial_inv = np.array([-initial[0], -initial[1], -initial[2], initial[3]])  
        relative = self.quat_multiply(current, initial_inv)
        return relative
        
    def rotation_matrix(self, quaternion):
        x=quaternion[0]
        y=quaternion[1]
        z=quaternion[2]
        w = quaternion[3]
        return np.array([
            [1 - 2 * (y**2 + z**2), 2 * (x*y - z * w), 2 *(x*z + y*w)],
            [2 *(x*y + z*w), 1 - 2 * (x**2 + z**2), 2 * (y*z - x*w)],
            [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 *(x**2+ y**2)]
        ])
    
    def save_data(self, points):
        filename = 'transformed_points.xyz'
        with open(filename, 'a') as file:
            for x, y, z in points:
                file.write(f"{x} {y} {z}\n")
                
    def transform_points(self, points, orientation):
        rotation_matrix = self.rotation_matrix(orientation)
        transformed_points = [np.dot(rotation_matrix, point) for point in points]
        return transformed_points
                
    # pointcloud2 message for rviz debugging of data 
    def create_pointcloud2(self, points):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'Frame3d'
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
    node = VerticalLidarMapper()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

