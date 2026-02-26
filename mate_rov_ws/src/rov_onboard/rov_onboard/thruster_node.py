"""
Thruster Node - Receives thruster commands from control laptop and drives motors.
Runs on the LattePanda (onboard computer).
Placeholder: replace serial/PWM logic with your actual motor driver code.
"""

import rclpy
from rclpy.node import Node
from rov_msgs.msg import ThrusterCommand


class ThrusterNode(Node):
    def __init__(self):
        super().__init__('thruster_node')

        self.declare_parameter('armed', False)
        self.armed = self.get_parameter('armed').value

        self.subscription = self.create_subscription(
            ThrusterCommand,
            '/rov/thruster_command',
            self.thruster_callback,
            10
        )

        self.get_logger().info('Thruster node started (disarmed by default)')

    def thruster_callback(self, msg):
        if not self.armed:
            return

        # TODO: Replace with actual motor driver commands (e.g. serial to Arduino/ESC)
        self.get_logger().debug(
            f'Thrusters: FL={msg.thruster_front_left:.2f} '
            f'FR={msg.thruster_front_right:.2f} '
            f'BL={msg.thruster_back_left:.2f} '
            f'BR={msg.thruster_back_right:.2f} '
            f'VL={msg.thruster_vertical_left:.2f} '
            f'VR={msg.thruster_vertical_right:.2f}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = ThrusterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
