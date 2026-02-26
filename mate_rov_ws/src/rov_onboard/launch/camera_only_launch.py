"""
Minimal launch file — camera node only.
Useful for testing just the camera stream before full system is ready.
Run this on the LattePanda.
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('rov_onboard'),
        'config',
        'onboard_params.yaml'
    )

    return LaunchDescription([
        Node(
            package='rov_onboard',
            executable='camera_node',
            name='camera_node',
            parameters=[config],
            output='screen',
        ),
    ])
