from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
import os

def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('KogRob-EtoE-NN-Driving'),
        'config', 'teleop_joy.yaml'
    )

    return LaunchDescription([
        Node(
            package='joy',
            executable='joy_node',
            name='joy_node',
            parameters=[config],
            output='screen'
        ),
        Node(
            package='teleop_twist_joy',
            executable='teleop_node',
            name='teleop_twist_joy_node',
            parameters=[config],
            output='screen'
        ),
        Node(
            package='KogRob-EtoE-NN-Driving',
            executable='joy_republisher',
            name='joy_republisher',
            output='screen'
        )
    ])