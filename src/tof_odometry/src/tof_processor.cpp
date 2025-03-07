#include <memory>
#include <cmath>
#include <mutex>
#include "rclcpp/rclcpp.hpp"
// Sensor Message
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
// Odometry Message
#include <nav_msgs/msg/odometry.hpp>
// Point Cloud Library
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <Eigen/Dense>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
using namespace std::chrono_literals;

class TOFProcessor : public rclcpp::Node
{
public:
    TOFProcessor() : Node("tof_processor")
    {
        // ToF Setup
        previous_cloud = nullptr;
        current_cloud = nullptr;
        this->declare_parameter("output_topic_num", "one");
        rclcpp::Parameter output_topic_num_param = this->get_parameter("output_topic_num");
        std::string output_depth_topic = "/depth_" + output_topic_num_param.as_string();
        output_odom_topic = "tof_odom_" + output_topic_num_param.as_string();
        std::string output_frame_id = "tof_" + output_topic_num_param.as_string();
        tof_subscription =
            this->create_subscription<sensor_msgs::msg::Image>(output_depth_topic, 10, std::bind(&TOFProcessor::tof_callback, this, std::placeholders::_1));
        
        // IMU Setup
        has_initial_orientation = false;
        imu_subscription =
            this->create_subscription<sensor_msgs::msg::Imu>("/bno055/imu", 10, std::bind(&TOFProcessor::imu_callback, this, std::placeholders::_1));
        
        // Processing Setup
        relative_transformation = Eigen::Matrix4d::Identity();
        global_transformation = Eigen::Matrix4d::Identity();
        ekf_transformation = Eigen::Matrix4d::Identity();
        has_filtered_odometry = false;
        ekf_subscription =
            this->create_subscription<nav_msgs::msg::Odometry>("/odometry/filtered", 10, std::bind(&TOFProcessor::ekf_callback, this, std::placeholders::_1));
        // Synchronizes ToF and IMU data to 10Hz
        timer_ = this->create_wall_timer(std::chrono::milliseconds(100), std::bind(&TOFProcessor::process_messages, this));
        // Publishers
        std::string output_pointcloud_topic = "/tof_cloud_" + output_topic_num_param.as_string();
        std::string output_rotated_topic = "/tof_rotated_" + output_topic_num_param.as_string();
        odometry_pub_ = this->create_publisher<nav_msgs::msg::Odometry>(output_odom_topic, 10);
        rotation_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(output_rotated_topic, 10);
        pointcloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(output_pointcloud_topic, 10);
        // Broadcaster for rotational transform to "/tf" for Rviz
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
    }
    
private:
    // IMU callback for IMU subscription
    void imu_callback(const std::shared_ptr<const sensor_msgs::msg::Imu> &msg)
    {
        tf2::Quaternion current_quat(msg->orientation.x, msg->orientation.y, msg->orientation.z, msg->orientation.w);
        
        // Store initial orientation
        if (!has_initial_orientation) {
            initial_orientation = current_quat;
            has_initial_orientation = true;
            RCLCPP_INFO(this->get_logger(), "Initial IMU orientation stored:\nx: %lf\ny: %lf\nz: %lf\nw: %lf\n", msg->orientation.x, msg->orientation.y, msg->orientation.z, msg->orientation.w);
            return;
        }
        // Stores current orientation
        current_orientation = current_quat;
    }
    
    // ToF callback for ToF subscription
    void tof_callback(const std::shared_ptr<const sensor_msgs::msg::Image> &msg)
    {
        /*
            Setup
        */
        std::lock_guard<std::mutex> cloud_lock(cloud_mutex_);
        current_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>());
        relative_transformation = Eigen::Matrix4d::Identity();
        
        /*
            Depth Image to 3D Point Cloud
        */
        current_cloud->header.frame_id = msg->header.frame_id;
        current_cloud->width = msg->width; // should be 100 pixels
        current_cloud->height = msg->height; // should be 100 pixels
        current_cloud->is_dense = false;
        
        float cx = msg->width / 2.0; // should be 50
        float cy = msg->height / 2.0; // should be 50
        
        for (uint32_t v = 0; v < msg->height; ++v)
        {
            for (uint32_t u = 0; u < msg->width; ++u)
            {
                /*
                    Depth Setting: 1-10
                        1 for 1mm
                        10 for 10mm
                    Setting 5:
                        Minimum: 1 = 5mm = 0.005m
                        Maximum: 255 = 1275mm = 1.275m 
                */
                // float depth = cv_ptr->image.at<float>(v, u) * 0.005;
                float depth = msg->data[v * 100 + u] * 0.005;

                if (std::isfinite(depth) && depth > 0.0)
                {
                    /*
                        Horizontal FOV: 70 deg, deltax = 70 deg / 100 pixels = 0.7 deg/pixel
                        Vertical FOV: 60 deg, deltay = 60 deg / 100 pixels = 0.6 deg/pixel
                        x = depth * tan (0.7 * pixel width * radian conversion)
                        y = depth * tan (0.6 * pixel height * radian conversion)
                    */
                    pcl::PointXYZ point;
                    point.x = depth * std::tan((u - cx) * 0.7 *3.14159/180); // 0.7 deg/pixel
                    point.y = depth * std::tan((v - cy) * 0.6 *3.14159/180); // 0.6 deg/pixel
                    point.z = depth; // should be on max setting which is 1cm (max 255cm for 8 bit)
                    current_cloud->points.push_back(point);
                }
            }
        }
        sensor_msgs::msg::PointCloud2 ros_cloud;
        pcl::toROSMsg(*current_cloud, ros_cloud);
        ros_cloud.header.stamp = this->now();
        ros_cloud.header.frame_id = current_cloud->header.frame_id;
        pointcloud_pub_->publish(ros_cloud);
    }
    
    // EKF callback for EKF subscription
    void ekf_callback(const std::shared_ptr<const nav_msgs::msg::Odometry> &msg){
        std::lock_guard<std::mutex> ekf_lock(ekf_mutex_);
        ekf_transformation = Eigen::Matrix4d::Identity();
        ekf_transformation(0, 3) = msg->pose.pose.position.x;
        ekf_transformation(1, 3) = msg->pose.pose.position.y;
        ekf_transformation(2, 3) = msg->pose.pose.position.z;
        tf2::Quaternion quat(
            msg->pose.pose.orientation.x,
            msg->pose.pose.orientation.y,
            msg->pose.pose.orientation.z,
            msg->pose.pose.orientation.w
        );
        tf2::Matrix3x3 tf2_rotation(quat);
        Eigen::Matrix3d rotation;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                rotation(i, j) = tf2_rotation[i][j];
            }
        }
        ekf_transformation.block<3, 3>(0, 0) = rotation;
        if (!has_filtered_odometry) has_filtered_odometry = true;
    }
    
    void process_messages(){
        
        std::lock_guard<std::mutex> cloud_lock(cloud_mutex_);
        if (!current_cloud) {
            RCLCPP_INFO(this->get_logger(), "Missing current cloud");
            return;
        }
        
        /*
            IMU Data Processing
        */
        
        if (current_orientation){
            RCLCPP_INFO(this->get_logger(), "Processing message at 10Hz");
        } else {
            RCLCPP_INFO(this->get_logger(), "Processing message, missing current orientation");
            return;
        }
        // Calculate rotational transform between current and initial orientation
        tf2::Quaternion rotation_difference = current_orientation * initial_orientation.inverse();
        rotation_difference.normalize();
        tf2::Matrix3x3 rotation_matrix(rotation_difference);
        
        /*
            Apply Rotational Transform
        */
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        transformed_cloud->header.frame_id = current_cloud->header.frame_id;
        transformed_cloud->width = current_cloud->width;
        transformed_cloud->height = current_cloud->height;
        transformed_cloud->is_dense = current_cloud->is_dense;
        transformed_cloud->points.resize(current_cloud->points.size());
        for (size_t i = 0; i < current_cloud->points.size(); ++i) {
            const pcl::PointXYZ &point = current_cloud->points[i];
            pcl::PointXYZ transformed_point;
            transformed_point.x = rotation_matrix[0][0] * point.x + rotation_matrix[0][1] * point.y + rotation_matrix[0][2] * point.z;
            transformed_point.y = rotation_matrix[1][0] * point.x + rotation_matrix[1][1] * point.y + rotation_matrix[1][2] * point.z;
            transformed_point.z = rotation_matrix[2][0] * point.x + rotation_matrix[2][1] * point.y + rotation_matrix[2][2] * point.z;
            transformed_cloud->points[i] = transformed_point;
        }
        
        sensor_msgs::msg::PointCloud2 ros_cloud;
        pcl::toROSMsg(*transformed_cloud, ros_cloud);
        ros_cloud.header.stamp = this->now();
        ros_cloud.header.frame_id = transformed_cloud->header.frame_id;
        rotation_pub_->publish(ros_cloud);
        
        /*
            ICP (Iterative Closest Point)
            - Source: transformed_cloud
            - Target: previous_cloud
        */
        if (previous_cloud) {
            pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
            icp.setInputSource(transformed_cloud);
            icp.setInputTarget(previous_cloud);
            pcl::PointCloud<pcl::PointXYZ> aligned_cloud;
            icp.align(aligned_cloud);
            if (icp.hasConverged()) {
                RCLCPP_INFO(this->get_logger(), "ICP converged. Fitness score: %f", icp.getFitnessScore());
                relative_transformation = icp.getFinalTransformation().cast<double>();
            } else {
                RCLCPP_INFO(this->get_logger(), "ICP did not converge");
                relative_transformation = Eigen::Matrix4d::Identity();
                return;
            }
        } else {
            previous_cloud = transformed_cloud;
            RCLCPP_INFO(this->get_logger(), "Missing previous cloud");
            return;
        }
        
        /*
            Updating and Publishing Transformation
        */
        std::lock_guard<std::mutex> ekf_lock(ekf_mutex_);
        previous_cloud = transformed_cloud;
        // if (has_filtered_odometry) global_transformation =  ekf_transformation; // TODO FIX
        global_transformation =  global_transformation * relative_transformation;
        publish_odometry(global_transformation, rotation_difference);
        
        /*
            Logging Messages
        */
        std::cout << "Relative Matrix: " << std::endl;
        std::cout << relative_transformation << std::endl;
        std::cout << "Global Matrix: " << std::endl;
        std::cout << global_transformation << std::endl;
        // Print relative translation and rotation
        Eigen::Vector3d translation = relative_transformation.block<3, 1>(0, 3);
        RCLCPP_INFO(this->get_logger(), "Translation (x,y,z): %f %f %f\n", translation.x(), translation.y(), translation.z());
        RCLCPP_INFO(this->get_logger(), "Rotation (x,y,z,w): %f %f %f %f\n", rotation_difference.x(), rotation_difference.y(), rotation_difference.z(), rotation_difference.w());
        //Print global translation and rotation
        Eigen::Vector3d global_translation = global_transformation.block<3, 1>(0, 3);
        RCLCPP_INFO(this->get_logger(), "Global Translation (x,y,z): %f %f %f\n", global_translation.x(), global_translation.y(), global_translation.z());
        
        /*
            Broadcasting Transform
        */
        geometry_msgs::msg::TransformStamped broadcast_msg;
        broadcast_msg.header.stamp = this->now();
        broadcast_msg.header.frame_id = "base_link";  // Parent frame
        broadcast_msg.child_frame_id = "imu_link";   // IMU frame
        broadcast_msg.transform.translation.x = translation.x();
        broadcast_msg.transform.translation.y = translation.y();
        broadcast_msg.transform.translation.z = translation.z();
        broadcast_msg.transform.rotation.x = rotation_difference.x();
        broadcast_msg.transform.rotation.y = rotation_difference.y();
        broadcast_msg.transform.rotation.z = rotation_difference.z();
        broadcast_msg.transform.rotation.w = rotation_difference.w();
        // Publish the transform
        tf_broadcaster_->sendTransform(broadcast_msg);
    }
    
    void publish_odometry(const Eigen::Matrix4d &transformation, const tf2::Quaternion &quaternion) {
        // Extract translation and rotation
        Eigen::Vector3d translation = transformation.block<3, 1>(0, 3);
        // Eigen::Matrix3d rotation_matrix = transformation.block<3, 3>(0, 0);
        // Eigen::Quaterniond quaternion(rotation_matrix);
        // Create Odometry message
        nav_msgs::msg::Odometry odometry_msg;
        odometry_msg.header.stamp = this->get_clock()->now();
        // TODO, change to parameterized
        odometry_msg.header.frame_id = "base_link";
        odometry_msg.child_frame_id = output_odom_topic;
        odometry_msg.pose.pose.position.x = translation.x();
        odometry_msg.pose.pose.position.y = translation.y();
        odometry_msg.pose.pose.position.z = translation.z();
        odometry_msg.pose.pose.orientation.x = quaternion.x();
        odometry_msg.pose.pose.orientation.y = quaternion.y();
        odometry_msg.pose.pose.orientation.z = quaternion.z();
        odometry_msg.pose.pose.orientation.w = quaternion.w();
        // Publish odometry
        odometry_pub_->publish(odometry_msg);
    }
    
    // IMU Setup
    bool has_initial_orientation;
    tf2::Quaternion initial_orientation;
    tf2::Quaternion current_orientation;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscription;
    // ToF Setup
    std::mutex cloud_mutex_;
    std::string output_odom_topic;
    pcl::PointCloud<pcl::PointXYZ>::Ptr previous_cloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr current_cloud;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr tof_subscription;
    // Processsing Setup
    bool has_filtered_odometry;
    std::mutex ekf_mutex_;
    Eigen::Matrix4d ekf_transformation;
    Eigen::Matrix4d relative_transformation;
    Eigen::Matrix4d global_transformation;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr ekf_subscription;
    rclcpp::TimerBase::SharedPtr timer_;
    // Publishing
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometry_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr rotation_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_pub_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TOFProcessor>());
  rclcpp::shutdown();
  return 0;
}