#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
import os
import sys
import select
import tty
import termios
from datetime import datetime
import threading
import concurrent.futures
from collections import deque
import re

_original_fd = sys.stdin.fileno() if sys.stdin.isatty() else None
original_terminal_settings = None
if _original_fd is not None:
    try:
        original_terminal_settings = termios.tcgetattr(_original_fd)
    except termios.error:
        original_terminal_settings = None



class ImageRecorderNode(Node):
    def __init__(self):
        super().__init__('image_recorder_node')

        self.declare_parameter('use_cmd_vel', False)
        self.use_cmd_vel = self.get_parameter('use_cmd_vel').get_parameter_value().bool_value

        # Determine the package's root directory.
        script_path = os.path.realpath(__file__)
        # Script is run in the install folder but we want to see data under src
        package_install_dir = os.path.dirname(os.path.dirname(script_path))
        package_source_dir = re.sub(r"/install/.*", "/src/KogRob-EtoE-NN-Driving", package_install_dir)
        self.output_dir = os.path.join(package_source_dir, "labelled_data")

        # Subscriptions
        self.image_topic = "/image_raw/compressed"
        self.velocity_topic = "/cmd_vel" if self.use_cmd_vel else "/joy_xy"

        self.is_recording = False
        self._recording_lock = threading.Lock()
        self.file_writer_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        try:
            os.makedirs(self.output_dir, exist_ok=True)
            self.get_logger().info(f"Saving images to: {os.path.abspath(self.output_dir)}")
        except OSError as e:
            self.get_logger().error(f"Failed to create output directory {self.output_dir}: {e}")
            rclpy.shutdown()
            sys.exit(1)

        # Buffer for velocity messages with arrival times
        self.vel_buffer = deque(maxlen=50)

        # Image buffer for downsampling ~10FPS
        self.image_buffer = []  # Stores (timestamp, image_msg)
        self.next_capture_time = None


        self.create_subscription(
            CompressedImage,
            self.image_topic,
            self.image_callback,
            qos_profile_sensor_data
        )

        if self.use_cmd_vel:
            self.create_subscription(
                Twist,
                self.velocity_topic,
                self.velocity_callback_cmd_vel,
                qos_profile_sensor_data
            )
            self.get_logger().info("Using /cmd_vel topic for velocity data.")
        else:
            self.create_subscription(
                Float32MultiArray,
                self.velocity_topic,
                self.velocity_callback_joy_xy,
                qos_profile_sensor_data
            )
            self.get_logger().info("Using /joy_xy topic for velocity data.")

        self.get_logger().info("Node initialized.")
        self.get_logger().info(f"Image: {self.image_topic}")
        self.get_logger().info(f"Velocity: {self.velocity_topic}")

        self.tty_out = None
        if sys.stdin.isatty():
            try:
                self.tty_out = open('/dev/tty', 'w')
            except Exception:
                self.tty_out = None

        if not self.tty_out:
            self.get_logger().warn("Cannot open /dev/tty; keyboard status prints disabled.")

        if sys.stdin.isatty() and original_terminal_settings:
            self.get_logger().info("Press 'r' to toggle recording. Press 'q' to quit.\n")
            self.keyboard_thread = threading.Thread(target=self.keyboard_listener)
            self.keyboard_thread.daemon = True
            self.keyboard_thread.start()
        else:
            self.get_logger().warn("stdin is not a TTY or terminal settings unavailable; keyboard controls disabled.")



    def velocity_callback_cmd_vel(self, msg: Twist):
        self.vel_buffer.append((self.get_clock().now(), msg))



    def velocity_callback_joy_xy(self, msg: Float32MultiArray):
        self.vel_buffer.append((self.get_clock().now(), msg))



    def image_callback(self, image_msg: CompressedImage):

        if not rclpy.ok():
            return
        
        now = self.get_clock().now()

        with self._recording_lock:
            if not self.is_recording:
                return

        self.get_logger().debug("Image received while recording!")

        # Buffer image
        self.image_buffer.append((now, image_msg))
        if len(self.image_buffer) > 5:
            self.image_buffer.pop(0)

        try:
            # First time setup
            if self.next_capture_time is None:
                self.next_capture_time = now + rclpy.duration.Duration(seconds=0.1)
                return

            # If not yet time to capture
            if now < self.next_capture_time:
                return

            # Find image closest to target time
            if not self.image_buffer: # Defensive check
                return
            closest_img = min(self.image_buffer, key=lambda x: abs((x[0] - self.next_capture_time).nanoseconds))
            closest_time, chosen_image_msg = closest_img

            # Find matching velocity
            if not self.vel_buffer:
                return

            best_vel_time, best_vel = min(
                self.vel_buffer,
                key=lambda item: abs((item[0] - closest_time).nanoseconds)
            )

            delta_sec = abs((best_vel_time - closest_time).nanoseconds) * 1e-9
            if delta_sec > 1.0:
                self.get_logger().warn(
                    f"Image (ROS time: {closest_time.nanoseconds*1e-9:.3f}) and best velocity (ROS time: {best_vel_time.nanoseconds*1e-9:.3f}) "
                    f"data too far apart: {delta_sec:.3f}s. Skipping image.\r\n"
                )
                return

            self._save_image_with_velocity(chosen_image_msg, best_vel, closest_time)

            # Update capture time
            self.next_capture_time = self.next_capture_time + rclpy.duration.Duration(seconds=0.1)

            # Clear buffer of old images
            self.image_buffer = [img for img in self.image_buffer if img[0] > self.next_capture_time - rclpy.duration.Duration(seconds=0.05)]
        except Exception as e:
            self.get_logger().error(f"Unexpected error in image_callback processing loop: {e}", exc_info=True)



    def _save_image_with_velocity(self, image_msg: CompressedImage, velocity_msg, image_ros_time: rclpy.time.Time):
        # Use the provided ROS timestamp of the image for the filename for data integrity
        dt_object = datetime.fromtimestamp(image_ros_time.nanoseconds / 1e9)
        timestamp_str = dt_object.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        ext = 'jpg' if 'jpeg' in (image_msg.format or '').lower() else 'png'


        epsilon = 1e-6
        suffix = ''
        if self.use_cmd_vel and isinstance(velocity_msg, Twist):
            x, y = velocity_msg.linear.x, velocity_msg.angular.z
        elif isinstance(velocity_msg, Float32MultiArray):
            data = velocity_msg.data
            x, y = (data[0], data[1]) if len(data) >= 2 else (0.0, 0.0)

        else: # Should not happen if subscriptions and parameters are correct
            self.get_logger().warn(
                f"Unexpected velocity message type ({type(velocity_msg)}) or invalid data "
                f"when trying to save image for ROS time {image_ros_time.nanoseconds*1e-9:.3f}. Skipping save.\n"
            )
            return
        
        if abs(x) < epsilon and abs(y) < epsilon: return
        xs = f"{x:.3f}".replace('.', 'p').replace('-', 'n')
        ys = f"{y:.3f}".replace('.', 'p').replace('-', 'n')
        suffix = f"X{xs}_Y{ys}"

        filename = f"{timestamp_str}_{suffix}.{ext}"
        path = os.path.join(self.output_dir, filename)
        try:
            self.file_writer_executor.submit(self._write_file, path, image_msg.data)
        except RuntimeError: # Raised when submitting to a closed executor
            # This is expected during shutdown, so no error log needed unless rclpy is still ok.
            if rclpy.ok():
                self.get_logger().error("Attempted to submit save task to a closed executor while rclpy is still OK.")
        # No specific logging for RuntimeError if rclpy is not ok, as it's expected.



    def _write_file(self, path, data):
        try:
            with open(path, 'wb') as f:
                f.write(data)
        except Exception as e:
            self.get_logger().error(f"Failed to save image {os.path.basename(path)}: {e}")



    def keyboard_listener(self):
        # This method is only called if _original_fd and original_terminal_settings are valid.
        try:
            tty.setraw(_original_fd)
            self._print_status() # Initial status
            while rclpy.ok():
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    if key == '\x03': # Ctrl+C character
                        self.get_logger().info("Ctrl+C pressed via keyboard listener. Shutting down...")
                        self._print_message("Ctrl+C pressed, quitting...")
                        rclpy.shutdown() # Initiate shutdown
                        break # Exit listener loop
                    elif key == 'r':
                        with self._recording_lock:
                            self.is_recording = not self.is_recording
                        self.get_logger().info(f"Recording {'STARTED' if self.is_recording else 'STOPPED'}")
                        self._print_status()
                    elif key == 'q':
                        self.get_logger().info("Quit key 'q' pressed. Shutting down...")
                        self._print_message("Quitting...")
                        rclpy.shutdown() # Initiate shutdown
                        break # Exit listener loop
        except Exception as e:
            # Log unexpected errors in the listener thread
            try:
                self.get_logger().error(f"Error in keyboard_listener: {e}", exc_info=True)
            except: # Fallback if logger itself is problematic during shutdown
                print(f"[ERROR] Critical error in keyboard_listener: {e}")
        finally:
            # Crucially, always restore terminal settings
            self.restore_terminal()



    def _print_status(self):
        if not self.tty_out: return
        status = 'STARTED' if self.is_recording else 'STOPPED' # Ensure self.is_recording is accessed safely if needed
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
            except Exception as e:
                # Try to log, but don't crash if logging fails during shutdown
                try:
                    self.get_logger().warn(f"Failed to restore terminal settings: {e}")
                except:
                    # If logger is not available (e.g. node fully shut down), print as a last resort.
                    print(f"[WARN] Failed to restore terminal settings (logger unavailable): {e}")



    def destroy_node(self):
        self.get_logger().info("Shutting down file writer executor...")
        self.file_writer_executor.shutdown(wait=True)
        self.restore_terminal()
        if self.tty_out:
            self.tty_out.write("\rShutdown complete.\n")
            self.tty_out.flush()
            self.tty_out.close() # Explicitly close the tty output stream
        super().destroy_node()




def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ImageRecorderNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node:
            node.get_logger().info("KeyboardInterrupt received, shutting down.")
        # else: # Not strictly necessary to print if node is None
            # print("KeyboardInterrupt received during node initialization, shutting down.")
    except Exception as e:
        if node:
            node.get_logger().error(f"Unhandled exception: {e}", exc_info=True)
        else:
            print(f"Unhandled exception during node creation: {e}")
    finally:
        if node:
            node.destroy_node()
        if rclpy.ok(): # Call shutdown only if not already shut down
            rclpy.shutdown()


if __name__ == '__main__':
    main()
