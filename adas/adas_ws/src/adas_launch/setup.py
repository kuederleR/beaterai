from setuptools import find_packages, setup

package_name = 'adas_launch'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch',
         ['launch/adas_pipeline.launch.py']),
    ],
    install_requires=['setuptools'],
)
