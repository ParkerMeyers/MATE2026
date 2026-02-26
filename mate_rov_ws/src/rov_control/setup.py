from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'rov_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'config'),
            glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='MATE ROV Team',
    maintainer_email='team@mate-rov.org',
    description='Control station nodes for MATE ROV (runs on laptop)',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_viewer_node = rov_control.camera_viewer_node:main',
            'gamepad_node = rov_control.gamepad_node:main',
            'dashboard_node = rov_control.dashboard_node:main',
        ],
    },
)
