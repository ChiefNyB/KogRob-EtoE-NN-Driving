#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import os
import re

class FlipImageTest(Node):
    def __init__(self):
        super().__init__('flip_image_test')

        # Adjust dataset path to ensure it points to the source folder
        if 'install' in os.path.dirname(__file__):
            dataset_path = re.sub(r"/install/.*", "/src/KogRob-EtoE-NN-Driving/labelled_data", os.path.dirname(__file__))
        else:
            dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'labelled_data'))

        image_paths = [os.path.join(root, file) 
                       for root, _, files in os.walk(dataset_path) 
                       for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not image_paths:
            self.get_logger().error("No images found in the dataset.")
            return

        # Use the first image for testing
        input_image_path = image_paths[0]
        output_image_path = os.path.splitext(input_image_path)[0] + "_test" + os.path.splitext(input_image_path)[1]
        original_copy_path = os.path.splitext(input_image_path)[0] + "_original" + os.path.splitext(input_image_path)[1]

        # Load the image
        image = cv2.imread(input_image_path)
        if image is None:
            self.get_logger().error(f"Could not read image: {input_image_path}")
            return

        # Save a copy of the original image
        cv2.imwrite(original_copy_path, image)
        self.get_logger().info(f"Original image copied to: {original_copy_path}")

        # Apply the flipping method (horizontal flip)
        flipped_image = image[:, ::-1, :]

        # Save the flipped image
        cv2.imwrite(output_image_path, flipped_image)
        self.get_logger().info(f"Flipped image saved to: {output_image_path}")

        # Simulate label flipping as in create_and_train_cnn
        filename = os.path.basename(input_image_path)
        filename_no_ext = os.path.splitext(filename)[0]
        parts = filename_no_ext.split('_')

        try:
            x_val = None
            y_val = None
            for part in parts:
                if part.startswith('X') and len(part) > 1:
                    x_str = part[1:].replace('p', '.').replace('n', '-')
                    x_val = float(x_str)
                elif part.startswith('Y') and len(part) > 1:
                    y_str = part[1:].replace('p', '.').replace('n', '-')
                    y_val = float(y_str)

            if x_val is not None and y_val is not None:
                flipped_x_val = -x_val  # Invert steering
                self.get_logger().info(f"Original labels: X={x_val}, Y={y_val}")
                self.get_logger().info(f"Flipped labels: X={flipped_x_val}, Y={y_val}")
            else:
                self.get_logger().warning(f"Could not parse valid X/Y values from filename: {filename}")
        except Exception as e:
            self.get_logger().error(f"Error parsing labels from filename: {filename}. Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = FlipImageTest()
    rclpy.spin_once(node, timeout_sec=0)  # Spin once to allow logging
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()