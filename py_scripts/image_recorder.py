#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray
import message_filters
import os
import sys
import select
import tty
import termios
from datetime import datetime
import threading
import concurrent.futures

# Store original terminal settings
original_terminal_settings = termios.tcgetattr(sys.stdin)

class ImageRecorderNode(Node):
    """
    A ROS2 node that records compressed images based on keyboard input and joystick values.

    Subscribes to:
        /image/compressed (sensor_msgs/CompressedImage): The compressed image stream. (Synchronized)
        /joy_xy (std_msgs/Float32MultiArray): Joystick X and Y values ([X, Y]).

    Functionality:
        - Press 'r' to toggle recording on/off.
        - When recording, saves images to 'labelled_data/' directory.
        - Filename format: YYYYMMDD_HHMMSS_ms_X<joy_x>_Y<joy_y>.jpg
        - Press 'q' or Ctrl+C to quit gracefully.
    """
    def __init__(self):
        super().__init__('image_recorder_node')

        # Parameters (optional, could be declared/fetched)
        self.output_dir = "labelled_data"
        self.image_topic = "/image_raw/compressed"
        self.joy_xy_topic = "/joy_xy" # Combined topic

        # Recording state variables
        self.is_recording = False
        self._recording_lock = threading.Lock() # Lock for thread-safe access to is_recording
        
        # New thread for file writing
        self.file_writer_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1) # Only need 1 worker for sequential writes

        # Create output directory if it doesn't exist
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            self.get_logger().info(f"Saving images to: {os.path.abspath(self.output_dir)}")
        except OSError as e:
            self.get_logger().error(f"Failed to create output directory {self.output_dir}: {e}")
            # Optionally exit or handle differently
            rclpy.shutdown()
            sys.exit(1)


        # --- Message Filters Setup ---
        # Create subscribers using message_filters
        self.image_sub = message_filters.Subscriber(self, CompressedImage, self.image_topic)
        self.joy_xy_sub = message_filters.Subscriber(self, Float32MultiArray, self.joy_xy_topic)

        # Create an ApproximateTimeSynchronizer
        # Adjust queue_size and slop as needed
        # queue_size: How many messages of each type to store before matching.
        # slop: Max time difference (in seconds) allowed between messages.
        self.time_synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.joy_xy_sub], queue_size=30, slop=0.1, allow_headerless=True

        )
        # Register the callback for synchronized messages
        self.time_synchronizer.registerCallback(self.synchronized_callback)
        # -----------------------------

        self.get_logger().info(f"Node initialized. Subscribed to:")
        self.get_logger().info(f"  Image: {self.image_topic}")
        self.get_logger().info(f"  Joy XY: {self.joy_xy_topic}") # Updated log message
        self.get_logger().info("Press 'r' to toggle recording. Press 'q' to quit.")

        # Start keyboard listener thread
        self.keyboard_thread = threading.Thread(target=self.keyboard_listener)
        self.keyboard_thread.daemon = True # Allows program to exit even if thread is running
        self.keyboard_thread.start()

    def save_image_async(self, filepath, data):
        """Function to be run in a separate thread for saving."""
        try:
            with open(filepath, 'wb') as f:
                f.write(data)
            # self.get_logger().debug(f"Saved image: {filename}") # Careful logging from threads
        except IOError as e:
            self.get_logger().error(f"Failed to save image {os.path.basename(filepath)}: {e}")
        except Exception as e:
            self.get_logger().error(f"An unexpected error occurred while saving image: {e}")

    def synchronized_callback(self, image_msg: CompressedImage, joy_msg: Float32MultiArray):
        """Callback function for processing synchronized image and joystick messages."""
        # --- Keep the recording check and data extraction in the main thread ---
        with self._recording_lock:
            if not self.is_recording:
                return

        joy_x = 0.0
        joy_y = 0.0
        if len(joy_msg.data) >= 2:
            joy_x = joy_msg.data[0]
            joy_y = joy_msg.data[1]
        else:
            self.get_logger().warn(f"Received synchronized Float32MultiArray with < 2 elements: {len(joy_msg.data)}. Using 0.0 for X/Y.")

        epsilon = 1e-6
        if abs(joy_x) < epsilon and abs(joy_y) < epsilon:
            self.get_logger().debug("Skipping image: Velocities near zero.")
            return

        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        joy_x_str = f"{joy_x:.3f}".replace('.', 'p').replace('-', 'n')
        joy_y_str = f"{joy_y:.3f}".replace('.', 'p').replace('-', 'n')

        extension = "jpg"
        if image_msg.format and "jpeg" in image_msg.format.lower():
            extension = "jpg"
        elif image_msg.format and "png" in image_msg.format.lower():
            extension = "png"

        filename = f"{timestamp}_X{joy_x_str}_Y{joy_y_str}.{extension}"
        filepath = os.path.join(self.output_dir, filename)
        image_data = image_msg.data # Copy data needed by the thread

        # --- Submit the file writing task to the executor ---
        self.file_writer_executor.submit(self.save_image_async, filepath, image_data)


    def keyboard_listener(self):
        """Listens for keyboard input ('r' to toggle recording, 'q' to quit)."""
        # Set terminal to raw mode to capture single key presses
        tty.setraw(sys.stdin.fileno())
        print("\r", end="") # Clear potential prompt artifacts

        while rclpy.ok():
            # Use select for non-blocking check of stdin
            if select.select([sys.stdin], [], [], 0.1)[0]: # Timeout of 0.1 seconds
                key = sys.stdin.read(1)
                if key == 'r':
                    with self._recording_lock:
                        self.is_recording = not self.is_recording
                        status = "STARTED" if self.is_recording else "STOPPED"
                        self.get_logger().info(f"Recording {status}")
                        print(f"\rRecording {status}. Press 'r' again to toggle, 'q' to quit.") # Print status to terminal
                elif key == 'q':
                    self.get_logger().info("Quit key pressed. Shutting down...")
                    print("\rQuitting...")

                    rclpy.shutdown()
                    break # Exit the listener loop
            # Small sleep to prevent high CPU usage if select has 0 timeout
            # time.sleep(0.01) # Already handled by select timeout

        # Restore terminal settings when loop exits
        self.restore_terminal()

    def restore_terminal(self):
        """Restores the terminal to its original settings."""
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, original_terminal_settings)
        print("\rTerminal settings restored.") # Print confirmation

    def destroy_node(self):
        """Clean up resources."""
        self.get_logger().info("Shutting down file writer executor...")
        self.file_writer_executor.shutdown(wait=True) # Wait for pending writes
        self.restore_terminal() # Ensure terminal is restored on shutdown
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ImageRecorderNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\rCtrl+C detected. Shutting down...")
    except Exception as e:
        if node:
            node.get_logger().error(f"Unhandled exception: {e}")
        else:
            print(f"Unhandled exception during node creation: {e}")
    finally:
        # Cleanup
        if node:
            node.destroy_node() # This includes restoring terminal settings
        if rclpy.ok():
             rclpy.shutdown()
        # Explicitly restore terminal just in case destroy_node wasn't called properly
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, original_terminal_settings)
        print("\rShutdown complete.")


if __name__ == '__main__':
    main()
