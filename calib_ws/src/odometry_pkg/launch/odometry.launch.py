from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('odometry_pkg'),
        'config',
        'params.yaml'
    )

    return LaunchDescription([
        Node(
            package='odometry_pkg',
            executable='visual_odom_node',
            name='visual_odom_node',
            output='screen',
            parameters=[config]
        ),
        Node(
            package='odometry_pkg',
            executable='lidar_odom_node',
            name='lidar_odom_node',
            output='screen',
            parameters=[config]
        )
    ])
