"""
Launch file for all control station nodes.
Run this on the control laptop.
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('rov_control'),
        'config',
        'control_params.yaml'
    )

    return LaunchDescription([
        Node(
            package='rov_control',
            executable='camera_viewer_node',
            name='camera_viewer_node',
            parameters=[config],
            output='screen',
        ),
        Node(
            package='rov_control',
            executable='gamepad_node',
            name='gamepad_node',
            parameters=[config],
            output='screen',
        ),
        Node(
            package='rov_control',
            executable='dashboard_node',
            name='dashboard_node',
            parameters=[config],
            output='screen',
        ),
    ])
