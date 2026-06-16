import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Header
from cv_bridge import CvBridge


class DepthNode(Node):
    def __init__(self):
        super().__init__('depth_node')

        self.declare_parameter('model_name',
                               'depth-anything/Depth-Anything-V3-Small')
        self.declare_parameter('device', 'cuda' if self._cuda_available() else 'cpu')
        self.declare_parameter('inference_stride', 2)
        self.declare_parameter('pointcloud_stride', 4)
        self.declare_parameter('max_depth', 80.0)
        self.declare_parameter('camera_frame_id', 'camera_link')

        model_name = self.get_parameter('model_name').value
        self._device = self.get_parameter('device').value
        self._inf_stride = self.get_parameter('inference_stride').value
        self._pc_stride = self.get_parameter('pointcloud_stride').value
        self._max_depth = self.get_parameter('max_depth').value
        self._frame_id = self.get_parameter('camera_frame_id').value

        self._frame_counter = 0
        self._bridge = CvBridge()
        self._camera_info = None
        self._model = None
        self._processor = None

        self._depth_pub = self.create_publisher(Image, '/depth/image_raw', 10)
        self._pc_pub = self.create_publisher(PointCloud2, '/depth/points', 10)

        self.create_subscription(
            Image, '/camera/image_raw', self._image_callback, 10)
        self.create_subscription(
            CameraInfo, '/camera/camera_info', self._camera_info_callback, 10)

        self._load_model(model_name)

    @staticmethod
    def _cuda_available():
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _load_model(self, model_name):
        self.get_logger().info(f'Loading depth model: {model_name}')
        try:
            from transformers import (AutoImageProcessor,
                                      AutoModelForDepthEstimation)
            self._processor = AutoImageProcessor.from_pretrained(model_name)
            self._model = AutoModelForDepthEstimation.from_pretrained(
                model_name)
            import torch
            self._model.to(self._device)
            self._model.eval()
            self.get_logger().info('Depth model loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')

    def _camera_info_callback(self, msg):
        self._camera_info = msg

    def _image_callback(self, msg):
        self._frame_counter += 1
        if self._frame_counter % self._inf_stride != 0:
            return

        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        depth = self._estimate_depth(rgb)
        if depth is None:
            return

        depth_msg = self._bridge.cv2_to_imgmsg(
            depth.astype(np.float32), encoding='32FC1')
        depth_msg.header = msg.header
        depth_msg.header.frame_id = self._frame_id
        self._depth_pub.publish(depth_msg)

        if self._camera_info is not None:
            pc_msg = self._depth_to_pointcloud2(depth, msg.header.stamp)
            if pc_msg is not None:
                self._pc_pub.publish(pc_msg)

    def _estimate_depth(self, rgb_image):
        if self._model is None or self._processor is None:
            return None
        try:
            import torch
            inputs = self._processor(
                images=rgb_image, return_tensors='pt')
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)

            if hasattr(outputs, 'predicted_depth'):
                depth = outputs.predicted_depth
            elif hasattr(outputs, 'depth'):
                depth = outputs.depth
            else:
                self.get_logger().error(
                    f'Unknown model output format: {type(outputs)}')
                return None

            depth = depth.squeeze().cpu().numpy()
            depth = cv2.resize(
                depth, (rgb_image.shape[1], rgb_image.shape[0]),
                interpolation=cv2.INTER_LINEAR)
            return depth
        except Exception as e:
            self.get_logger().error(f'Depth inference failed: {e}')
            return None

    def _depth_to_pointcloud2(self, depth, stamp):
        if self._camera_info is None:
            return None

        h, w = depth.shape
        k = self._camera_info.k
        fx, fy, cx, cy = k[0], k[4], k[2], k[5]

        v, u = np.mgrid[0:h:self._pc_stride, 0:w:self._pc_stride]
        z = depth[v, u].astype(np.float32)

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        mask = (z > 0.1) & (z < self._max_depth) & np.isfinite(z)

        points = np.stack([
            x[mask].ravel(),
            y[mask].ravel(),
            z[mask].ravel(),
        ], axis=-1)

        if points.shape[0] == 0:
            return None

        msg = PointCloud2()
        msg.header = Header(stamp=stamp, frame_id=self._frame_id)
        msg.height = 1
        msg.width = points.shape[0]
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        msg.data = points.tobytes()
        return msg

    def destroy_node(self):
        if self._model is not None:
            self._model.cpu()
            del self._model
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DepthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
