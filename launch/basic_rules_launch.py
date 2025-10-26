from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('project1'),
        'config',
        'params.yaml'
    )

    return LaunchDescription([
        Node(
            package='project1',
            executable='basic_rules',
            name='basic_rules',
            parameters=[config],
            output='screen'
        )
    ])