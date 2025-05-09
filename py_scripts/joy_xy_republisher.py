import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Joy
import random

class JoyxyRepublisher(Node):
    def __init__(self):
        super().__init__('joy_republisher')

        self.declare_parameter("axis_linear.x", 1)
        self.declare_parameter("axis_angular.yaw", 0)

        self.axis_linear_x = self.get_parameter("axis_linear.x").get_parameter_value().integer_value
        self.axis_angular_yaw = self.get_parameter("axis_angular.yaw").get_parameter_value().integer_value

        self.publisher_ = self.create_publisher(Joy, 'joy', 10)
        timer_period = 0.05  # seconds
        self.timer = self.create_timer(timer_period, self.publish_joy)
        self.axes = [0.0, 0.0, 0.0 ,0.0, 1.0, 1.0, 0.0, 0.0]


        self.subscribtion = self.create_subscription(Float32MultiArray, 'joy_xy', self.listener_callback, 10)

        self.get_logger().info('JoyxyRepublisher node started.')

    def publish_joy(self):
        msg = Joy()
        msg.axes = self.axes
        self.publisher_.publish(msg)
    
    def listener_callback(self, msg):
        self.axes[self.axis_angular_yaw] = msg.data[0]
        self.axes[self.axis_linear_x] = msg.data[1]

def main(args=None):
    rclpy.init(args=args)
    node = JoyxyRepublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()