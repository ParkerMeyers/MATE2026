from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'rov_onboard'

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
    description='Onboard nodes for MATE ROV (runs on LattePanda)',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_node = rov_onboard.camera_node:main',
            'thruster_node = rov_onboard.thruster_node:main',
            'sensor_node = rov_onboard.sensor_node:main',
            'status_node = rov_onboard.status_node:main',
        ],
    },
)
