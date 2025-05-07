#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray  # For /joy_xy
from geometry_msgs.msg import Twist        # For /cmd_vel
import message_filters
import os
import sys
import select
import tty
import termios
from datetime import datetime
import threading
import concurrent.futures

# Store original terminal settings (use fileno)
_original_fd = sys.stdin.fileno() if sys.stdin.isatty() else None
original_terminal_settings = None
if _original_fd is not None:
    try:
        original_terminal_settings = termios.tcgetattr(_original_fd)
    except termios.error:
        original_terminal_settings = None

class ImageRecorderNode(Node):
    """
    A ROS2 node that records compressed images based on keyboard input and velocity data.

    Subscribes to:
        /image_raw/compressed (sensor_msgs/CompressedImage): The compressed image stream. (Synchronized)
        One of:
            /joy_xy (std_msgs/Float32MultiArray): Joystick X and Y values ([X, Y]).
            /cmd_vel (geometry_msgs/Twist): Command velocity messages.

    Functionality:
        - Press 'r' to toggle recording on/off.
        - When recording, saves images to 'labelled_data/' directory.
        - Filename format depends on the velocity source.
        - Press 'q' or Ctrl+C to quit gracefully.
    """
    def __init__(self):
        super().__init__('image_recorder_node')

        # Declare parameter for choosing velocity topic
        self.declare_parameter('use_cmd_vel', False)  # Default to False (use /joy_xy)
        self.use_cmd_vel = self.get_parameter('use_cmd_vel').get_parameter_value().bool_value

        # Configuration
        self.output_dir = "labelled_data"
        self.image_topic = "/image_raw/compressed"

        # Recording state variables
        self.is_recording = False
        self._recording_lock = threading.Lock()

        # File writer executor
        self.file_writer_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        # Create output directory
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            self.get_logger().info(f"Saving images to: {os.path.abspath(self.output_dir)}")
        except OSError as e:
            self.get_logger().error(f"Failed to create output directory {self.output_dir}: {e}")
            rclpy.shutdown()
            sys.exit(1)

        # Message filters with sensor_data QoS
        self.image_sub = message_filters.Subscriber(
            self,
            CompressedImage,
            self.image_topic,
            qos_profile=qos_profile_sensor_data
        )
        if self.use_cmd_vel:
            self.velocity_topic = "/cmd_vel"
            self.velocity_sub = message_filters.Subscriber(
                self,
                Twist,
                self.velocity_topic,
                qos_profile=qos_profile_sensor_data
            )
            self.get_logger().info(f"Using /cmd_vel topic for velocity data.")
        else:
            self.velocity_topic = "/joy_xy"
            self.velocity_sub = message_filters.Subscriber(
                self,
                Float32MultiArray,
                self.velocity_topic,
                qos_profile=qos_profile_sensor_data
            )
            self.get_logger().info(f"Using /joy_xy topic for velocity data.")

        # Synchronizer: holds ~0.5s of messages at 30fps
        self.time_synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.velocity_sub],
            queue_size=15,
            slop=0.2,
            allow_headerless=True
        )
        self.time_synchronizer.registerCallback(self.synchronized_callback)

        self.get_logger().info("Node initialized. Subscribed to:")
        self.get_logger().info(f"  Image: {self.image_topic}")
        self.get_logger().info(f"  Velocity: {self.velocity_topic}")

        # Terminal output handle (/dev/tty) for status
        self.tty_out = None
        if sys.stdin.isatty():
            try:
                self.tty_out = open('/dev/tty', 'w')
            except Exception:
                self.tty_out = None
        if not self.tty_out:
            self.get_logger().warn("Cannot open /dev/tty; keyboard status prints disabled.")

        # Keyboard listener
        if sys.stdin.isatty() and original_terminal_settings:
            self.get_logger().info("Press 'r' to toggle recording. Press 'q' to quit.")
            self.keyboard_thread = threading.Thread(target=self.keyboard_listener)
            self.keyboard_thread.daemon = True
            self.keyboard_thread.start()
        else:
            self.get_logger().warn("stdin is not a TTY or terminal settings unavailable; keyboard controls disabled.")

    def save_image_async(self, filepath, data):
        try:
            with open(filepath, 'wb') as f:
                f.write(data)
        except Exception as e:
            self.get_logger().error(f"Failed to save image {os.path.basename(filepath)}: {e}")

    def synchronized_callback(self, image_msg: CompressedImage, velocity_msg):
        with self._recording_lock:
            if not self.is_recording:
                return

        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        ext = 'jpg' if 'jpeg' in (image_msg.format or '').lower() else 'png'

        epsilon = 1e-6
        suffix = ''
        if self.use_cmd_vel:
            if not isinstance(velocity_msg, Twist): return
            lx, az = velocity_msg.linear.x, velocity_msg.angular.z
            if abs(lx) < epsilon and abs(az) < epsilon: return
            lin = f"{lx:.3f}".replace('.', 'p').replace('-', 'n')
            ang = f"{az:.3f}".replace('.', 'p').replace('-', 'n')
            suffix = f"LinX{lin}_AngZ{ang}"
        else:
            if not isinstance(velocity_msg, Float32MultiArray): return
            data = velocity_msg.data
            x, y = (data[0], data[1]) if len(data)>=2 else (0.0, 0.0)
            if abs(x) < epsilon and abs(y) < epsilon: return
            xs = f"{x:.3f}".replace('.', 'p').replace('-', 'n')
            ys = f"{y:.3f}".replace('.', 'p').replace('-', 'n')
            suffix = f"X{xs}_Y{ys}"

        filename = f"{timestamp}_{suffix}.{ext}"
        path = os.path.join(self.output_dir, filename)
        self.file_writer_executor.submit(self.save_image_async, path, image_msg.data)

    def keyboard_listener(self):
        # raw mode
        tty.setraw(_original_fd)
        # initial print
        self._print_status()
        while rclpy.ok():
            if select.select([sys.stdin], [], [], 0.1)[0]:
                key = sys.stdin.read(1)
                if key == 'r':
                    with self._recording_lock:
                        self.is_recording = not self.is_recording
                    self.get_logger().info(f"Recording {'STARTED' if self.is_recording else 'STOPPED'}")
                    self._print_status()
                elif key == 'q':
                    self.get_logger().info("Quit key pressed. Shutting down...")
                    self._print_message("Quitting...")
                    rclpy.shutdown()
                    break
        self.restore_terminal()

    def _print_status(self):
        if not self.tty_out: return
        status = 'STARTED' if self.is_recording else 'STOPPED'
        self.tty_out.write(f"\rRecording {status}. Press 'r' to toggle, 'q' to quit.")
        self.tty_out.flush()

    def _print_message(self, msg):
        if not self.tty_out: return
        self.tty_out.write(f"\r{msg:<80}")
        self.tty_out.flush()

    def restore_terminal(self):
        if _original_fd is not None and original_terminal_settings:
            try:
                termios.tcsetattr(_original_fd, termios.TCSADRAIN, original_terminal_settings)
            except Exception:
                pass

    def destroy_node(self):
        self.get_logger().info("Shutting down file writer executor...")
        self.file_writer_executor.shutdown(wait=True)
        self.restore_terminal()
        # final message
        if self.tty_out:
            self.tty_out.write("\rShutdown complete.\n")
            self.tty_out.flush()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ImageRecorderNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        if node:
            node.get_logger().error(f"Unhandled exception: {e}")
        else:
            print(f"Unhandled exception during node creation: {e}")
    finally:
        if node:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
