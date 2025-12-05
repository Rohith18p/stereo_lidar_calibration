from setuptools import find_packages, setup
import os
from glob import glob
package_name = 'odometry_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nail',
    maintainer_email='rohith18p@gmail.com',
    description='Camera and Lidar Odometry',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'visual_odom_node = odometry_pkg.visual_odometry:main',
            'lidar_odom_node = odometry_pkg.lidar_odometry:main',
            'extrinsic_node = odometry_pkg.calibration:main',
        ],
    },
)