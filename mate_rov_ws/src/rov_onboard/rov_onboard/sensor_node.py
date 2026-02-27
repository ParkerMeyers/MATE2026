"""
Sensor Node - Reads sensors and publishes data to the control laptop.
Runs on the LattePanda (onboard computer).
Placeholder: replace with actual sensor reading logic (I2C, serial, etc.).
"""

import rclpy
from rclpy.node import Node
from rov_msgs.msg import SensorData


class SensorNode(Node):
    def __init__(self):
        super().__init__('sensor_node')

        self.declare_parameter('publish_rate', 30.0)
        publish_rate = self.get_parameter('publish_rate').value

        self.publisher = self.create_publisher(SensorData, '/rov/sensor_data', 10)

        timer_period = 1.0 / publish_rate
        self.timer = self.create_timer(timer_period, self.publish_sensors)

        self.get_logger().info(f'Sensor node started at {publish_rate} Hz')

    def publish_sensors(self):
        msg = SensorData()

        # TODO: Replace with actual sensor readings
        msg.depth = 0.0
        msg.temperature = 0.0
        msg.heading = 0.0
        msg.pitch = 0.0
        msg.roll = 0.0
        msg.battery_voltage = 0.0

        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SensorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
