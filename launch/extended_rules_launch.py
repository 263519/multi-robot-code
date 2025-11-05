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

    # FOR DEBUGGING IF THE PARAMS ARE LOADED PROPERLY FROM THE YAML FILE
    # print(f"Loading params from: {config}")  # <--- ADD THIS
    # if not os.path.exists(config):
    #     print(f"WARNING: Config file NOT FOUND: {config}")  # <--- ADD THIS

    return LaunchDescription([
        Node(
            package='project1',
            executable='extended_rules',
            name='extended_rules', # default name. only used if the node did not set its own name in the code (class initialisation)
            parameters=[config],
            output='screen'
        )
    ])