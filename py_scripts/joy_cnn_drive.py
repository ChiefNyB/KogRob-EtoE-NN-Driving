from __future__ import annotations

import os
import select
import sys
import termios
import tty
from threading import Lock
from typing import List, Optional, Sequence, Type, Any

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float32MultiArray

# ────────────────────────── Default parameters ─────────────────────────
DEFAULTS = {
    "model_path": (
        "/home/mibalazs/ros2_ws/KR_project_ws/src/"
        "KogRob-EtoE-NN-Driving/network_model/best_model.keras"
    ),
    "image_width": 200,
    "image_height": 66,
    "camera_topic": "image_raw",
    "publish_topic": "joy_xy",
    "constant_speed": 0.1,
    "force_cpu": False,
    # GUI / visualisation
    "display_camera": True,
    "display_activations": True,
    "activation_stride": 5,      # update every N frames
    "activation_tile_height": 73,  # px – *height* now controls scale
    "activation_max_width": 99999,  # stop drawing if we run out of space
}


# ────────────────────────────── Node class ─────────────────────────────
class JoyCNNDrive(Node):
    """Drive the TurtleBot using a steering‐only CNN model with live GUI."""

    def __init__(self):
        super().__init__("joy_cnn_drive_node")

        # ─────────────── read/declare parameters ────────────────
        for k, v in DEFAULTS.items():
            self.declare_parameter(k, v)
        p = lambda k: self.get_parameter(k).get_parameter_value()

        self.img_w = p("image_width").integer_value
        self.img_h = p("image_height").integer_value
        self.constant_speed = float(p("constant_speed").double_value)
        camera_topic = p("camera_topic").string_value
        publish_topic = p("publish_topic").string_value
        model_path = p("model_path").string_value
        force_cpu = p("force_cpu").bool_value
        self.display_camera = p("display_camera").bool_value
        self.show_acts = self.display_camera and p("display_activations").bool_value
        self.act_stride = max(1, p("activation_stride").integer_value)
        self.tile_h = max(40, p("activation_tile_height").integer_value)
        self.max_act_w = max(200, p("activation_max_width").integer_value)

        # ───────────── TensorFlow / model setup ────────────────
        if force_cpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        import tensorflow as tf  # delayed import

        try:
            for gpu in tf.config.list_physical_devices("GPU"):
                tf.config.experimental.set_memory_growth(gpu, True)
            if tf.config.list_physical_devices("GPU"):
                self.get_logger().info("GPU detected; memory‑growth on")
            else:
                self.get_logger().info("Running on CPU")
        except Exception as e:  # pragma: no cover
            self.get_logger().warn(f"GPU init failed → CPU only: {e}")
            tf.config.set_visible_devices([], "GPU")

        from tensorflow.keras.layers import Conv2D  # type: ignore
        from tensorflow.keras.models import load_model  # type: ignore

        self.get_logger().info(f"Loading model: {model_path}")
        self.model = load_model(model_path, compile=False)
        self.get_logger().info("Model loaded ✔")

        # Conv layers for activation preview
        self._act_models: List[tf.keras.Model] = []
        self._act_images: List[Optional[Any]] = []
        if self.show_acts:
            conv_layers = self._pick_conv_layers(self.model.layers, Conv2D)
            if not conv_layers:
                self.show_acts = False
                self.get_logger().warn("No Conv2D layers found – overlay disabled")
            else:
                self.get_logger().info(
                    f"Visualising layers: {[l.name for l in conv_layers]}")
                for l in conv_layers:
                    self._act_models.append(tf.keras.Model(self.model.inputs, l.output))
                    self._act_images.append(None)

        # ───────────── ROS I/O setup ────────────────
        msg_type: Type[Image] | Type[CompressedImage]
        self._is_compressed: bool
        if camera_topic.endswith("compressed"):
            msg_type, self._is_compressed = CompressedImage, True
        else:
            msg_type, self._is_compressed = Image, False

        self.bridge = CvBridge()
        self.sub = self.create_subscription(msg_type, camera_topic, self._image_cb, 10)
        self.pub = self.create_publisher(Float32MultiArray, publish_topic, 10)

        # ───────────── state / GUI initialisation ────────────────
        self.running = True
        self._frame_lock = Lock()
        self._frame: Optional[Any] = None  # latest BGR for GUI
        self._frame_idx = 0

        if self.display_camera:
            import cv2  # type: ignore

            cv2.namedWindow("drive_cam")
            cv2.setMouseCallback("drive_cam", self._mouse_cb)
            self.create_timer(0.03, self._gui_timer)

        self._setup_tty()
        self.get_logger().info("Ready – click button or press 's' to RUN/STOP")

    # ─────────────────── utilities ──────────────────────────
    @staticmethod
    def _pick_conv_layers(layers: Sequence, Conv2D):
        convs = [l for l in layers if isinstance(l, Conv2D)]
        return convs[:2]  # Always return the first 2 Conv2D layers if they exist

    # ───────────── image callback ─────────────
    def _image_cb(self, msg):
        import cv2, numpy as np  # type: ignore
        from tensorflow.keras.preprocessing.image import img_to_array  # type: ignore

        frame = (
            self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            if self._is_compressed
            else self.bridge.imgmsg_to_cv2(msg, "bgr8")
        )

        # share to GUI
        if self.display_camera:
            with self._frame_lock:
                self._frame = frame.copy()

        # preprocess
        img = cv2.resize(frame, (self.img_w, self.img_h))
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, 0)

        # predict
        x_val = float(self.model.predict(img, verbose=0).flatten()[0])
        x_val = float(np.clip(x_val, -1.0, 1.0)) if self.running else 0.0
        y_val = self.constant_speed if self.running else 0.0
        self.pub.publish(Float32MultiArray(data=[x_val, y_val]))

        # activations (every Nth frame)
        if self.show_acts and self._frame_idx % self.act_stride == 0:
            self._update_activations(img)
        self._frame_idx += 1

    # ───────────── activation helper ─────────────
    def _update_activations(self, net_in):
        import cv2, numpy as np  # type: ignore

        def montage(fmap, max_maps=24, grid=(4, 5)):
            fm = fmap[0]
            c = min(fm.shape[-1], max_maps)
            fm = fm[:, :, :c]
            fm -= fm.min(axis=(0, 1), keepdims=True)
            denom = fm.max(axis=(0, 1), keepdims=True)
            denom[denom == 0] = 1
            fm = (fm / denom * 255).astype(np.uint8)
            tiles = []
            for i in range(grid[0]):
                row = []
                for j in range(grid[1]):
                    idx = i * grid[1] + j
                    if idx < c:
                        tile = cv2.resize(fm[:, :, idx], (fm.shape[1], fm.shape[0]))
                    else:
                        tile = np.zeros_like(fm[:, :, 0])
                    row.append(tile)
                tiles.append(np.hstack(row))
            m = cv2.applyColorMap(np.vstack(tiles), cv2.COLORMAP_JET)
            # scale so height == tile_h
            scale = self.tile_h / m.shape[0]
            new_w = int(m.shape[1] * scale)
            return cv2.resize(m, (new_w, self.tile_h))

        for k, mdl in enumerate(self._act_models):
            act = mdl.predict(net_in, verbose=0)
            self._act_images[k] = montage(act)

    # ───────────── GUI timer ─────────────
    def _gui_timer(self):
        import cv2  # type: ignore

        with self._frame_lock:
            frame = self._frame.copy() if self._frame is not None else None
            acts = list(self._act_images) if self.show_acts else []
        if frame is None:
            return

        # overlay activations safely
        x_off = 5
        for img in acts:
            if img is None:
                continue
            h, w = img.shape[:2]
            if x_off + w + 5 > frame.shape[1]:
                break  # no space left
            frame[5 : 5 + h, x_off : x_off + w] = img
            x_off += w + 5

        # RUN/STOP button
        h, w, _ = frame.shape
        radius = 22
        cx, cy = w - radius - 10, h - radius - 10
        color = (0, 255, 0) if not self.running else (0, 0, 255)
        cv2.circle(frame, (cx, cy), radius, color, -1)
        label = "RUN" if not self.running else "STOP"
        cv2.putText(
            frame,
            label,
            (cx - 18, cy + 5) if not self.running else (cx - 20, cy + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            rclpy.shutdown()
        elif key == ord("s"):
            self._toggle()
        cv2.imshow("drive_cam", frame)

    # ───────────── mouse / keyboard helpers ─────────────
    def _toggle(self):
        self.running = not self.running
        self.get_logger().info("▶ RUN" if self.running else "■ STOP")

    def _mouse_cb(self, event, x, y, *_):
        if event != 1 or self._frame is None:
            return
        h, w, _ = self._frame.shape
        cx, cy, radius = w - 22 - 10, h - 22 - 10, 22
        if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
            self._toggle()

    def _setup_tty(self):
        try:
            self._orig_attr = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            self._tty_ok = True
            self.create_timer(0.05, self._tty_timer)
        except Exception:
            self._tty_ok = False

    def _tty_timer(self):
        if select.select([sys.stdin], [], [], 0)[0]:
            if sys.stdin.read(1).lower() == "s":
                self._toggle()

    # ───────────── cleanup ─────────────
    def destroy_node(self):
        if getattr(self, "_tty_ok", False):
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._orig_attr)
        if self.display_camera:
            import cv2  # type: ignore

            cv2.destroyAllWindows()
        super().destroy_node()


# ─────────────────────────────── main ─────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = JoyCNNDrive()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
