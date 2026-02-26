"""
Gamepad Node - Reads gamepad/joystick input and publishes thruster commands.
Runs on the control laptop.
Requires pygame: pip3 install pygame
"""

import rclpy
from rclpy.node import Node
from rov_msgs.msg import ThrusterCommand

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


class GamepadNode(Node):
    def __init__(self):
        super().__init__('gamepad_node')

        self.declare_parameter('publish_rate', 50.0)
        self.declare_parameter('deadzone', 0.1)
        self.declare_parameter('max_power', 1.0)

        publish_rate = self.get_parameter('publish_rate').value
        self.deadzone = self.get_parameter('deadzone').value
        self.max_power = self.get_parameter('max_power').value

        self.publisher = self.create_publisher(ThrusterCommand, '/rov/thruster_command', 10)

        if not PYGAME_AVAILABLE:
            self.get_logger().error('pygame not installed. Run: pip3 install pygame')
            return

        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            self.get_logger().warn('No gamepad detected — plug in a controller and restart')
            return

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        self.get_logger().info(f'Gamepad connected: {self.joystick.get_name()}')

        timer_period = 1.0 / publish_rate
        self.timer = self.create_timer(timer_period, self.read_gamepad)

    def apply_deadzone(self, value):
        if abs(value) < self.deadzone:
            return 0.0
        return value * self.max_power

    def read_gamepad(self):
        pygame.event.pump()

        # Standard mapping: left stick = strafe, right stick = yaw, triggers = vertical
        # Adjust axis indices to match your specific controller
        forward = -self.apply_deadzone(self.joystick.get_axis(1))   # Left stick Y
        strafe = self.apply_deadzone(self.joystick.get_axis(0))     # Left stick X
        yaw = self.apply_deadzone(self.joystick.get_axis(3))        # Right stick X
        vertical = self.apply_deadzone(self.joystick.get_axis(4))   # Right trigger - Left trigger

        # Convert to individual thruster values (basic mixing)
        msg = ThrusterCommand()
        msg.thruster_front_left = float(forward + strafe + yaw)
        msg.thruster_front_right = float(forward - strafe - yaw)
        msg.thruster_back_left = float(forward - strafe + yaw)
        msg.thruster_back_right = float(forward + strafe - yaw)
        msg.thruster_vertical_left = float(vertical)
        msg.thruster_vertical_right = float(vertical)

        # Clamp all values to [-1.0, 1.0]
        for attr in ['thruster_front_left', 'thruster_front_right',
                      'thruster_back_left', 'thruster_back_right',
                      'thruster_vertical_left', 'thruster_vertical_right']:
            val = getattr(msg, attr)
            setattr(msg, attr, max(-1.0, min(1.0, val)))

        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = GamepadNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
