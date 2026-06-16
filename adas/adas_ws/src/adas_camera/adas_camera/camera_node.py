import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        self.declare_parameter('device_path', '/dev/video0')
        self.declare_parameter('width', 1280)
        self.declare_parameter('height', 800)
        self.declare_parameter('fps', 30)
        self.declare_parameter('camera_frame_id', 'camera_link')

        device_path = self.get_parameter('device_path').value
        width = self.get_parameter('width').value
        height = self.get_parameter('height').value
        target_fps = self.get_parameter('fps').value
        self._frame_id = self.get_parameter('camera_frame_id').value

        self._bridge = CvBridge()
        self._cap = None
        self._camera_info = self._build_camera_info(width, height)

        self._image_pub = self.create_publisher(Image, '/camera/image_raw', 10)
        self._info_pub = self.create_publisher(CameraInfo, '/camera/camera_info', 10)

        self._cap = cv2.VideoCapture(device_path)
        if self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self._cap.set(cv2.CAP_PROP_FPS, target_fps)
            actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.get_logger().info(
                f'Camera opened: {device_path} ({actual_w}x{actual_h})')
            if actual_w != width or actual_h != height:
                self._camera_info = self._build_camera_info(actual_w, actual_h)
        else:
            self.get_logger().error(f'Failed to open camera: {device_path}')

        self._timer = self.create_timer(1.0 / target_fps, self._capture_and_publish)

    def _build_camera_info(self, width, height):
        info = CameraInfo()
        info.header.frame_id = self._frame_id
        info.width = width
        info.height = height
        info.distortion_model = 'plumb_bob'
        fx = fy = max(width, height) * 0.7
        cx = width / 2.0
        cy = height / 2.0
        info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        return info

    def _capture_and_publish(self):
        if self._cap is None or not self._cap.isOpened():
            return

        ret, frame = self._cap.read()
        if not ret:
            self.get_logger().warning('Frame capture failed, reopening camera...')
            self._cap.release()
            self._cap = None
            return

        stamp = self.get_clock().now().to_msg()

        ros_image = self._bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        ros_image.header.stamp = stamp
        ros_image.header.frame_id = self._frame_id
        self._image_pub.publish(ros_image)

        self._camera_info.header.stamp = stamp
        self._camera_info.header.frame_id = self._frame_id
        self._info_pub.publish(self._camera_info)

    def destroy_node(self):
        if self._cap is not None:
            self._cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
