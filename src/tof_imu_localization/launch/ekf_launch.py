from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Locate the configuration file
    config_path = os.path.join(
        get_package_share_directory('tof_imu_localization'),
        'config',
        'ekf_config.yaml'
    )

    # Define the EKF node
    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_localization_node',
        output='screen',
        parameters=[config_path]
    )

    # Return the launch description
    return LaunchDescription([
        ekf_node
    ])