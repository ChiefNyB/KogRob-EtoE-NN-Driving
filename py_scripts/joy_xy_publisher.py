import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Joy
import random

class JoyxyPublisher(Node):
    def __init__(self):
        super().__init__('joy_publisher')
        self.publisher_ = self.create_publisher(Float32MultiArray, 'joy_xy', 10)
        timer_period = 0.05  # seconds
        self.timer = self.create_timer(timer_period, self.publish_joy_xy)

        self.get_logger().info('JoyxyPublisher node started.')

    def publish_joy_xy(self):
        msg = Float32MultiArray()
        msg.data = [0.5, 0.5]
        self.publisher_.publish(msg)
    

def main(args=None):
    rclpy.init(args=args)
    node = JoyxyPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()