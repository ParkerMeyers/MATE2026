"""
Launch file for all onboard ROV nodes.
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
        Node(
            package='rov_onboard',
            executable='thruster_node',
            name='thruster_node',
            parameters=[config],
            output='screen',
        ),
        Node(
            package='rov_onboard',
            executable='sensor_node',
            name='sensor_node',
            parameters=[config],
            output='screen',
        ),
        Node(
            package='rov_onboard',
            executable='status_node',
            name='status_node',
            parameters=[config],
            output='screen',
        ),
    ])
