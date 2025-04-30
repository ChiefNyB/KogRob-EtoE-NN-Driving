import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Joy
import random

class JoyRepublisher(Node):
    def __init__(self):
        super().__init__('joy_republisher')
        self.publisher_ = self.create_publisher(Float32MultiArray, 'joy_xy', 10)
        timer_period = 0.05  # seconds
        self.timer = self.create_timer(timer_period, self.publish_joy_xy)
        self.axes = []


        self.subscribtion = self.create_subscription(Joy, 'joy', self.listener_callback, 10)

        self.get_logger().info('JoyRepublisher node started.')

    def publish_joy_xy(self):
        msg = Float32MultiArray()
        msg.data = self.axes
        self.publisher_.publish(msg)
    
    def listener_callback(self, msg):
        self.axes = msg.axes[0:2]

def main(args=None):
    rclpy.init(args=args)
    node = JoyRepublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()