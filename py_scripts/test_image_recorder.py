#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import message_filters
from geometry_msgs.msg import Twist # Import Twist message type
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
        /cmd_vel (geometry_msgs/Twist): The command velocity messages. (Synchronized)

    Functionality:
        - Press 'r' to toggle recording on/off.
        - When recording, saves images to 'labelled_data/' directory.
        - Filename format: YYYYMMDD_HHMMSS_ms_LinX<linear_x>_AngZ<angular_z>.jpg
        - Press 'q' or Ctrl+C to quit gracefully.
    """
    def __init__(self):
        super().__init__('image_recorder_node')

        # Parameters (optional, could be declared/fetched)
        self.output_dir = "labelled_data"
        self.image_topic = "/image_raw/compressed"
        self.cmd_vel_topic = "/cmd_vel" # Changed topic

        # Recording state variables
        self.is_recording = False
        self._recording_lock = threading.Lock() # Lock for thread-safe access to is_recording

        # New thread pool executor for file writing
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
        self.cmd_vel_sub = message_filters.Subscriber(self, Twist, self.cmd_vel_topic) # Use Twist message type

        # Create an ApproximateTimeSynchronizer
        # Adjust queue_size and slop as needed
        # queue_size: How many messages of each type to store before matching.
        # slop: Max time difference (in seconds) allowed between messages.
        self.time_synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.cmd_vel_sub], queue_size=15, slop=0.1, allow_headerless=True # Allow messages without headers
        )
        # Register the callback for synchronized messages
        self.time_synchronizer.registerCallback(self.synchronized_callback)
        # -----------------------------

        self.get_logger().info(f"Node initialized. Subscribed to:")
        self.get_logger().info(f"  Image: {self.image_topic}")
        self.get_logger().info(f"  Cmd Vel: {self.cmd_vel_topic}") # Updated log message
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
            # self.get_logger().debug(f"Saved image: {os.path.basename(filepath)}") # Uncomment if needed
        except IOError as e:
            self.get_logger().error(f"Failed to save image {os.path.basename(filepath)}: {e}")
        except Exception as e:
            self.get_logger().error(f"An unexpected error occurred while saving image {os.path.basename(filepath)}: {e}")

    def synchronized_callback(self, image_msg: CompressedImage, cmd_vel_msg: Twist):
        """Callback function for processing synchronized image and cmd_vel messages."""
        with self._recording_lock:
            if not self.is_recording:
                return # Do nothing if not recording

        # Extract linear.x and angular.z from the Twist message
        # These are typically the most relevant for ground robot control
        linear_x = cmd_vel_msg.linear.x
        angular_z = cmd_vel_msg.angular.z

        # --- Add check for non-zero velocity values ---
        # Example using tolerance (adjust epsilon as needed)
        epsilon = 1e-6
        if abs(linear_x) < epsilon and abs(angular_z) < epsilon:
            self.get_logger().debug("Skipping image: Velocities near zero.")
            return


        # Construct filename
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S_%f")[:-3] # YYYYMMDD_HHMMSS_ms
        lin_x_str = f"{linear_x:.3f}".replace('.', 'p').replace('-', 'n') # Replace decimal point and negative sign
        ang_z_str = f"{angular_z:.3f}".replace('.', 'p').replace('-', 'n') # Replace decimal point and negative sign

        # Determine file extension based on format (default to jpg)
        extension = "jpg"
        if image_msg.format and "jpeg" in image_msg.format.lower():
             extension = "jpg"
        elif image_msg.format and "png" in image_msg.format.lower():
             extension = "png"
        # Add more formats if needed

        filename = f"{timestamp}_LinX{lin_x_str}_AngZ{ang_z_str}.{extension}" # Updated filename format
        filepath = os.path.join(self.output_dir, filename)

        # --- Submit the file writing task to the executor ---
        image_data = image_msg.data # Copy data needed by the thread
        self.file_writer_executor.submit(self.save_image_async, filepath, image_data)
        # Callback returns quickly now


    def keyboard_listener(self):
        """Listens for keyboard input ('r' to toggle recording, 'q' to quit)."""
        # Set terminal to raw mode to capture single key presses
        tty.setraw(sys.stdin.fileno())
        print("\r", end="") # Clear potential prompt artifacts
        # Print initial status without newline to allow overwriting
        print(f"\rRecording {'STARTED' if self.is_recording else 'STOPPED'}. Press 'r' again to toggle, 'q' to quit.", end="")
        sys.stdout.flush() # Ensure it's displayed immediately

        while rclpy.ok():
            # Use select for non-blocking check of stdin
            if select.select([sys.stdin], [], [], 0.1)[0]: # Timeout of 0.1 seconds
                key = sys.stdin.read(1)
                if key == 'r':
                    new_status_str = ""
                    with self._recording_lock:
                        self.is_recording = not self.is_recording
                        status = "STARTED" if self.is_recording else "STOPPED"
                        new_status_str = f"Recording {status}. Press 'r' again to toggle, 'q' to quit."
                        self.get_logger().info(f"Recording {status}")
                    # Print status update outside the lock
                    print(f"\r{new_status_str:<80}", end="") # Pad with spaces to clear previous line
                    sys.stdout.flush()
                elif key == 'q':
                    self.get_logger().info("Quit key pressed. Shutting down...")
                    print("\rQuitting...                                        ") # Clear line
                    sys.stdout.flush()
                    rclpy.shutdown()
                    break # Exit the listener loop
            # Small sleep to prevent high CPU usage if select has 0 timeout
            # time.sleep(0.01) # Already handled by select timeout

        # Restore terminal settings when loop exits
        self.restore_terminal()

    def restore_terminal(self):
        """Restores the terminal to its original settings."""
        # Check if stdin is a tty before attempting to restore
        if sys.stdin.isatty():
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, original_terminal_settings)
            print("\rTerminal settings restored.") # Print confirmation
        else:
            self.get_logger().warn("stdin is not a TTY, cannot restore terminal settings.")

    def destroy_node(self):
        """Clean up resources."""
        self.get_logger().info("Shutting down node...")
        # Ensure keyboard listener thread knows to exit if shutdown initiated elsewhere
        # rclpy.shutdown() # Usually called externally or by keyboard listener

        # Shutdown executor gracefully
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
        elif sys.stdin.isatty(): # Restore terminal if node creation failed but stdin is tty
             termios.tcsetattr(sys.stdin, termios.TCSADRAIN, original_terminal_settings)
        if rclpy.ok(): # Ensure shutdown happens
             rclpy.shutdown()
        print("\rShutdown complete.")


if __name__ == '__main__':
    main()
