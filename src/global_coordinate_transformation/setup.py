from setuptools import find_packages, setup

package_name = 'global_coordinate_transformation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Hassan',
    maintainer_email='almasmoumh@gmail.com',
    description='Transforms vertical lidar scans to their corresponding 3D position given a reliable Odom source',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'global_coordinate_transformation = global_coordinate_transformation.global_coordinate_transformation:main',
        ],
    },
)

