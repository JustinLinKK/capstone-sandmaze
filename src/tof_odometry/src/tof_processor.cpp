#include <memory>
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
#include <tf2_ros/transform_broadcaster.h>
using namespace std::chrono_literals;

class TOFProcessor : public rclcpp::Node
{
public:
    TOFProcessor() : Node("tof_processor")
    {
        this->declare_parameter("output_topic_num", "one");
        rclcpp::Parameter output_topic_num_param = this->get_parameter("output_topic_num");
        std::string output_pointcloud_topic = "/cloud_" + output_topic_num_param.as_string();
        std::string output_odom_topic = "tof_odom_" + output_topic_num_param.as_string();
        std::string output_frame_id = "tof_" + output_topic_num_param.as_string();
        // Subscribers
        imu_subscription =
            this->create_subscription<sensor_msgs::msg::Imu>("/bno055/imu", 10, std::bind(&TOFProcessor::imu_callback, this, std::placeholders::_1));
        tof_subscription =
            this->create_subscription<sensor_msgs::msg::PointCloud2>(output_pointcloud_topic, 10, std::bind(&TOFProcessor::tof_callback, this, std::placeholders::_1));
        // Publishers
        odometry_pub_ = this->create_publisher<nav_msgs::msg::Odometry>(output_odom_topic, 10);
        // Broadcast
        timer_ = this->create_wall_timer(
            500ms, std::bind(&TOFProcessor::publish_transform, this));
        // Setup
        previous_cloud = nullptr;
        global_transformation = Eigen::Matrix4d::Identity();
    }
    
private:
    // IMU callback for IMU subscription
    void imu_callback(const std::shared_ptr<const sensor_msgs::msg::Imu> &msg)
    {
        RCLCPP_INFO(this->get_logger(), "Received IMU data:");
        (void) msg;   
    }
    // ToF callback for ToF subscription
    void tof_callback(const std::shared_ptr<const sensor_msgs::msg::PointCloud2> &msg)
    {
        /*
            Setup
        */
        pcl::PointCloud<pcl::PointXYZ>::Ptr current_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*msg, *current_cloud); //convert PointCloud2 to PointXYZ for PCL
        Eigen::Matrix4d relative_transformation = Eigen::Matrix4d::Identity();
        
        /*
            Filters
        */
        
        // Use voxel grid filter
        // pcl::PointCloud<pcl::PointXYZ>::Ptr voxel_cloud = downsample_cloud(current_cloud);
        // Apply noise filter
        pcl::PointCloud<pcl::PointXYZ>::Ptr processed_cloud = filter_outliers(current_cloud);
        
        
        /*
            TODO: Add initial guess for orientation/position based on IMU data
        */
        
        /*
            ICP (Iterative Closest Point)
            - Source: current_cloud or voxel_cloud or processed_cloud
            - Target: previous_cloud
        */
        if (previous_cloud) {
            pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
            icp.setInputSource(processed_cloud); // changes based on filters applied
            icp.setInputTarget(previous_cloud);
            pcl::PointCloud<pcl::PointXYZ> aligned_cloud;
            icp.align(aligned_cloud);
            if (icp.hasConverged()) {
                RCLCPP_INFO(this->get_logger(), "ICP converged. Fitness score: %f", icp.getFitnessScore());
                relative_transformation = icp.getFinalTransformation().cast<double>();
            } else {
                RCLCPP_INFO(this->get_logger(), "ICP did not converge");
                relative_transformation = Eigen::Matrix4d::Identity();
            }
        }

        /*
            Updating
        */
        previous_cloud = processed_cloud; // changes based on filters applied
        global_transformation =  global_transformation * relative_transformation;
        publish_odometry(global_transformation);
        
        /*
            Logging Messages
            
        */
        // // Print ToF and point cloud data
        // static int iterations = 0;
        // if (!iterations){
        //     RCLCPP_INFO(this->get_logger(), "Received ToF data:");
        //     RCLCPP_INFO(this->get_logger(), "Dimensions: [x=%u, y=%u]",
        //         msg->height, msg->width);
        //     for (int i=0; i<10; i++){
        //         uint8_t item = msg->data[i];
        //         RCLCPP_INFO(this->get_logger(), "Data: %u", item);
        //     }
        //     RCLCPP_INFO(this->get_logger(), "PointCloud has %zu points", current_cloud->points.size());
        //     for (const auto &point : current_cloud->points) {
        //         RCLCPP_INFO(this->get_logger(), "Point: x=%f, y=%f, z=%f", point.x, point.y, point.z);
        //     }
        //     RCLCPP_INFO(this->get_logger(), "PointCloud received with %zu points", current_cloud->size());
        // }
        // iterations++;
        // std::this_thread::sleep_for(std::chrono::milliseconds(5000));
        // Print transformation matrices
        std::cout << "Relative Matrix: " << std::endl;
        std::cout << relative_transformation << std::endl;
        std::cout << "Global Matrix: " << std::endl;
        std::cout << global_transformation << std::endl;
        // Print relative translation and rotation
        Eigen::Vector3d translation = relative_transformation.block<3, 1>(0, 3);
        Eigen::Matrix3d rotation_matrix = relative_transformation.block<3, 3>(0, 0);
        Eigen::Quaterniond quaternion(rotation_matrix);
        RCLCPP_INFO(this->get_logger(), "Translation (x,y,z): %f %f %f\n", translation.x(), translation.y(), translation.z());
        RCLCPP_INFO(this->get_logger(), "Rotation (x,y,z,w): %f %f %f %f\n", quaternion.x(), quaternion.y(), quaternion.z(), quaternion.w());
        //Print global translation and rotation
        Eigen::Vector3d global_translation = global_transformation.block<3, 1>(0, 3);
        RCLCPP_INFO(this->get_logger(), "Global Translation (x,y,z): %f %f %f\n", global_translation.x(), global_translation.y(), global_translation.z());
        // //Print global translation and rotation accumulated by adding relative values
        // static Eigen::Vector3d global_translation_alt = global_transformation.block<3, 1>(0, 3);
        // Eigen::Vector3d relative_translation = relative_transformation.block<3, 1>(0, 3);
        // global_translation_new = global_translation_new + relative_translation;
        // RCLCPP_INFO(this->get_logger(), "Global Translation Alternate(x,y,z): %f %f %f\n", global_translation_alt.x(), global_translation_alt.y(), global_translation_alt.z());          

    }
    
    void publish_odometry(const Eigen::Matrix4d &transformation) {
        // Extract translation and rotation
        Eigen::Vector3d translation = transformation.block<3, 1>(0, 3);
        Eigen::Matrix3d rotation_matrix = transformation.block<3, 3>(0, 0);
        Eigen::Quaterniond quaternion(rotation_matrix);
        // Create Odometry message
        nav_msgs::msg::Odometry odometry_msg;
        odometry_msg.header.stamp = this->get_clock()->now();
        // TODO, change to parameterized
        odometry_msg.header.frame_id = "odom";
        odometry_msg.child_frame_id = "tof_one";
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
    
    void publish_transform() {
        geometry_msgs::msg::TransformStamped transform;

        transform.header.stamp = this->get_clock()->now();
        // TODO, change to parameterized
        transform.header.frame_id = "odom";
        transform.child_frame_id = "tof_one";
        
        Eigen::Matrix4d transformation = global_transformation;

        transform.transform.translation.x = transformation(0, 3);
        transform.transform.translation.y = transformation(1, 3);
        transform.transform.translation.z = transformation(2, 3);

        Eigen::Matrix3d rotation_matrix = transformation.block<3, 3>(0, 0);
        Eigen::Quaterniond quaternion(rotation_matrix);

        transform.transform.rotation.x = quaternion.x();
        transform.transform.rotation.y = quaternion.y();
        transform.transform.rotation.z = quaternion.z();
        transform.transform.rotation.w = quaternion.w();

        tf_broadcaster->sendTransform(transform);
    }
    
    // Voxel Grid Filter
    pcl::PointCloud<pcl::PointXYZ>::Ptr downsample_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
        voxel_filter.setInputCloud(input_cloud);
        voxel_filter.setLeafSize(0.05, 0.05, 0.05);  // Adjust resolution
        voxel_filter.filter(*filtered_cloud);
        return filtered_cloud;
    }
    
    // Noise Filtering
    pcl::PointCloud<pcl::PointXYZ>::Ptr filter_outliers(pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(input_cloud);
        sor.setMeanK(50);
        sor.setStddevMulThresh(1.0);
        sor.filter(*filtered_cloud);
        return filtered_cloud;
    }
    
    
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscription;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr tof_subscription;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometry_pub_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr previous_cloud;
    Eigen::Matrix4d global_transformation;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster =
        std::make_unique<tf2_ros::TransformBroadcaster>(this);
    rclcpp::TimerBase::SharedPtr timer_;
    
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TOFProcessor>());
  rclcpp::shutdown();
  return 0;
}