"""
Camera Viewer Node - Subscribes to the compressed camera feed from the ROV
and displays it in an OpenCV window on the control laptop.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np


class CameraViewerNode(Node):
    def __init__(self):
        super().__init__('camera_viewer_node')

        self.declare_parameter('window_name', 'ROV Camera Feed')
        self.window_name = self.get_parameter('window_name').value

        self.subscription = self.create_subscription(
            CompressedImage,
            '/rov/camera/image_compressed',
            self.image_callback,
            10
        )

        self.frame_count = 0
        self.get_logger().info('Camera viewer started — waiting for frames...')

    def image_callback(self, msg):
        # Decode compressed image
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            self.get_logger().warn('Failed to decode frame')
            return

        # Display the frame
        cv2.imshow(self.window_name, frame)
        cv2.waitKey(1)

        self.frame_count += 1
        if self.frame_count % 150 == 0:
            self.get_logger().info(f'Received {self.frame_count} frames')

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraViewerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
