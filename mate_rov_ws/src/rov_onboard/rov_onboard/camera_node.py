"""
Camera Node - Captures USB camera frames and publishes them as ROS 2 image messages.
Runs on the LattePanda (onboard computer).
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        # Declare parameters
        self.declare_parameter('camera_index', 0)
        self.declare_parameter('frame_width', 640)
        self.declare_parameter('frame_height', 480)
        self.declare_parameter('fps', 30)
        self.declare_parameter('publish_compressed', True)
        self.declare_parameter('jpeg_quality', 50)

        # Get parameters
        camera_index = self.get_parameter('camera_index').value
        frame_width = self.get_parameter('frame_width').value
        frame_height = self.get_parameter('frame_height').value
        fps = self.get_parameter('fps').value
        self.publish_compressed = self.get_parameter('publish_compressed').value
        self.jpeg_quality = self.get_parameter('jpeg_quality').value

        # Construct a GStreamer pipeline string
        # This grabs the MJPEG stream, decodes it, and converts it to BGR for OpenCV
        pipeline = (
            f"v4l2src device=/dev/video{camera_index} io-mode=2 do-timestamp=true ! "
            f"image/jpeg, width={frame_width}, height={frame_height}, framerate={fps}/1 ! "
            f"jpegdec ! videoconvert ! "
            f"appsink sync=false max-buffers=1 drop=true"
        )
        
        # Initialize camera using GStreamer backend
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

        if not self.cap.isOpened():
            self.get_logger().error(f'Failed to open camera with GStreamer on /dev/video{camera_index}')
            raise RuntimeError(f'Cannot open camera {camera_index}')
            
        # (Note: You do not need the .set() commands for width/height/fps when using GStreamer 
        # because the pipeline string handles the negotiation automatically.)

        if not self.cap.isOpened():
            self.get_logger().error(f'Failed to open camera at index {camera_index}')
            raise RuntimeError(f'Cannot open camera {camera_index}')

        # CRITICAL: Force MJPEG format BEFORE setting resolution/FPS
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        # Set desired resolution and FPS
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        # Log the actual values the camera accepted
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.get_logger().info(
            f'Camera opened: requested {frame_width}x{frame_height}@{fps}fps, '
            f'actual {actual_w}x{actual_h}@{actual_fps:.1f}fps (index {camera_index})'
        )

        # Publishers
        self.bridge = CvBridge()
        self.image_pub = self.create_publisher(Image, '/rov/camera/image_raw', 10)

        if self.publish_compressed:
            self.compressed_pub = self.create_publisher(
                CompressedImage, '/rov/camera/image_compressed', 10
            )

        # Timer to capture frames
        timer_period = 1.0 / fps
        self.timer = self.create_timer(timer_period, self.publish_frame)
        self.frame_count = 0

    def publish_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('Failed to capture frame')
            return

        # Publish raw image
        img_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.header.frame_id = 'camera_link'
        self.image_pub.publish(img_msg)

        # Publish compressed image (much better for network streaming)
        if self.publish_compressed:
            compressed_msg = CompressedImage()
            compressed_msg.header = img_msg.header
            compressed_msg.format = 'jpeg'
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            _, encoded = cv2.imencode('.jpg', frame, encode_param)
            compressed_msg.data = encoded.tobytes()
            self.compressed_pub.publish(compressed_msg)

        self.frame_count += 1
        if self.frame_count % 150 == 0:
            self.get_logger().info(f'Published {self.frame_count} frames')

    def destroy_node(self):
        self.cap.release()
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
