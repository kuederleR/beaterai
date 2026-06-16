from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='adas_camera',
            executable='camera_node',
            name='camera_node',
            parameters=[{
                'device_path': '/dev/video0',
                'width': 1280,
                'height': 800,
                'fps': 30,
                'camera_frame_id': 'camera_link',
            }],
            output='screen',
        ),
        Node(
            package='adas_depth',
            executable='depth_node',
            name='depth_node',
            parameters=[{
                'model_name': 'depth-anything/Depth-Anything-V3-Small',
                'device': 'cuda',
                'inference_stride': 2,
                'pointcloud_stride': 4,
                'max_depth': 80.0,
                'camera_frame_id': 'camera_link',
            }],
            output='screen',
        ),
    ])
