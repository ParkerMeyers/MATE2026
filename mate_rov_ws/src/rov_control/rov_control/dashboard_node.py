"""
Dashboard Node - Subscribes to ROV status and sensor data, prints to terminal.
Runs on the control laptop.
Can be extended to a full GUI later (e.g. with PyQt5 or tkinter).
"""

import rclpy
from rclpy.node import Node
from rov_msgs.msg import SensorData, RovStatus


class DashboardNode(Node):
    def __init__(self):
        super().__init__('dashboard_node')

        self.sensor_sub = self.create_subscription(
            SensorData,
            '/rov/sensor_data',
            self.sensor_callback,
            10
        )

        self.status_sub = self.create_subscription(
            RovStatus,
            '/rov/status',
            self.status_callback,
            10
        )

        self.latest_sensor = None
        self.latest_status = None

        self.declare_parameter('display_rate', 2.0)
        display_rate = self.get_parameter('display_rate').value
        self.timer = self.create_timer(1.0 / display_rate, self.display)

        self.get_logger().info('Dashboard node started — waiting for data...')

    def sensor_callback(self, msg):
        self.latest_sensor = msg

    def status_callback(self, msg):
        self.latest_status = msg

    def display(self):
        lines = ['\n========== ROV DASHBOARD ==========']

        if self.latest_status:
            s = self.latest_status
            lines.append(f'  Armed:    {s.armed}')
            lines.append(f'  Mode:     {s.mode}')
            lines.append(f'  Leak:     {s.leak_detected}')
            lines.append(f'  CPU Temp: {s.cpu_temperature:.1f} °C')
            lines.append(f'  Uptime:   {s.uptime_seconds:.0f} s')
        else:
            lines.append('  Status: No data yet')

        lines.append('-----------------------------------')

        if self.latest_sensor:
            d = self.latest_sensor
            lines.append(f'  Depth:    {d.depth:.2f} m')
            lines.append(f'  Temp:     {d.temperature:.1f} °C')
            lines.append(f'  Heading:  {d.heading:.1f}°')
            lines.append(f'  Pitch:    {d.pitch:.1f}°')
            lines.append(f'  Roll:     {d.roll:.1f}°')
            lines.append(f'  Battery:  {d.battery_voltage:.2f} V')
        else:
            lines.append('  Sensors: No data yet')

        lines.append('===================================')
        self.get_logger().info('\n'.join(lines))


def main(args=None):
    rclpy.init(args=args)
    node = DashboardNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
