# KogRob: Robot control with end-to-end neural network
 
This project aims to implement a turtlebot3 driven by a CNN which reads images of the robot's camera, and outputs velocity commands to control the turtlebot3.
The foundation of the concepts and the network architecture was [this](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/) NVIDIA project.
For this project ROS2 Jazzy and Gazebo Harmonic were used.

Created by:
- Miklós Balázs
- Kristóf Bányi
- Ádám Szakmári
- Barnabás Nyuli


## 1. Install and setup

For use please install ROS2 Jazzy desktop version and Gazebo Harmonic.


Dependencies (beyond ROS2 and Gazebo):
- MOGI turtlebot3 repositories
- Python3
- Some python packages (see later)
- ROS2 joystick interface package

You can add these with cloning the following repositories into the src folder of the workspace:

```bash
git clone -b mogi-ros2 https://github.com/MOGI-ROS/turtlebot3
git clone -b new_gazebo https://github.com/MOGI-ROS/turtlebot3_simulations
```
And installing the packages below using apt:
```bash
sudo apt install python3-pip
sudo apt install pipx
sudo apt install ros-jazzy-joy-linux
sudo apt install ros-jazzy-teleop-twist-joy
sudo apt install ros-jazzy-message-filters
```

Set up a Python virtual enviroment, as described [here](https://github.com/MOGI-ROS/Week-1-8-Cognitive-robotics?tab=readme-ov-file#line-following)
and afterwards install these python packages:
```bash
pip install tensorflow==2.18.0
pip install imutils
pip install scikit-learn
pip install opencv-python
pip install matplotlib
pip install numpy==1.26.4
```

Because the CNN training is a standalone python script (not a ROS2 node), it is necessary to make it executable, by navigating to the ```py_scripts``` directory within the package and running the following command:
```bash
chmod +x create_and_train_cnn.py
```

Also don't forget to source the setup (workspace path may be different):

```bash
source ~/ros2_project/install/setup.bash
```

For GPU accelerated training process CUDA and cuDNN Library are required (optional).

Finally, after installing all dependencies, run the ```colcon build``` command in the root of the workspace.


## 2. How to use


### Controller
Manual control of the turtlebot with joystick can be started using the launch file:
```bash
ros2 launch KogRob-EtoE-NN-Driving joy_teleop_manual.launch.py
```
This converts the /joy topic of the controller into twist messages in the /cmd_vel topic, so it can drive the robot. It also publishes a simple version of the /joy topic, called /joy_xy. This topic only contains a list of the x and y coordinates of the joystick. The range of the coordinates is from -1.0 to 1.0. The /joy_xy topic makes it easier to communicate with the neural network. Publishing this topic when manually driving helps with labeling in the teaching process.\
You can also drive the robot straight from the /joy_xy topic with the launch file:
```bash
ros2 launch KogRob-EtoE-NN-Driving joy_teleop.launch.py
```
This launch file converts the coordinates in the topic to joy and the twist values and publishes it in /cmd_vel. You can also configure both the joystick teleops in ```config/teleop_joy.yaml```. This file configures mainly the speed of the robot and the joystick selection.\
You can test the control of the robot with the prepared worlds in the ```turtlebot3_gazebo``` package from the ```turtlebot3_simulations``` repository. An example of a test world and the turtlebot3:
```bash
ros2 launch turtlebot3_gazebo turtlebot3_house.launch.py
```

### Labelled data acquisition

To collect training data, recorded images paired with the corresponding joystick commands are saved. The `image_recorder` node is responsible for doing so. 

This node subscribes to the `/image_raw/compressed` topic (from the camera) and the `/joy_xy` topic (from the `joy_teleop_manual.launch.py` or a similar source).
When recording is active, the node receives images (downsampled to 10 FPS) and velocity commands in queues and pairs the closest ones together based on their arrival of the messages (ROS time).
If there are no velocity command matches for an image within a specified time, the image will not be saved. 
Also images will only be saved only if either the X or Y joystick value (or both) is non-zero to avoid saving images when the robot is stationary.

The training images are saved under the `labelled_data` folder within the package with a name similar to this: `20250521_201012_401_Xn0p500_Y0p860.jpg`
Here the first part shows the record date and time when the image was captured, and the numbers after `X` and `Y` represent the normalized angular (left-right) and linear (forward-backward) speed.
For easy file handling, the `.` and `-` symbols are exchanged for `p` and `n`. So in the example above an image was captured with `X: -0.500` and `Y: 0.860` values received on the joystick's topic.

The node can be started by running the following command:

```bash
ros2 run KogRob-EtoE-NN-Driving image_recorder
```
You can control the recording by:
*   Press the `r` key in the terminal where `image_recorder` is running to **start** recording.
*   Press `r` again to **stop** recording. You can toggle recording on and off as needed while driving the robot.
*   Press `q` or `Ctrl+C` to **quit** the recorder node gracefully.



The ```image_recorder``` node can be used in an altered mode, when listening to the ```\cmd_vel``` topic instead of ```\joy_xy``` . It is intended for testing and debug purposes only:
```bash
ros2 run KogRob-EtoE-NN-Driving image_recorder --ros-args -p use_cmd_vel:=True
```

### Neural network creation and teaching

The origin of the CNN model is NVIDIA's [DAVE-2](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/), with the slight modification of outputting linear and angular speed, instead of giving only the reciprical of the turning radius.
The network has an input of 200x66 pixels and X and Y outputs.

If a correctly labelled training dataset exists in the directory mentioned above, the CNN can be created and trained by navigating to the `py_scripts` directory and running the script:

```bash
python3 create_and_train_cnn.py
```
The process can be monitored in the terminal, and in case an error happens it is also printed here.
After a completed training procedure, the test evalution graphs pop-up showing the values of testfitting. The network structure is as follows:

```bash
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param   ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ layer_normalization             │ (None, 66, 200, 3)     │             6 │
│ (LayerNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d (Conv2D)                 │ (None, 31, 98, 24)     │         1,824 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout)               │ (None, 31, 98, 24)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_1 (Conv2D)               │ (None, 14, 47, 36)     │        21,636 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_1 (Dropout)             │ (None, 14, 47, 36)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_2 (Conv2D)               │ (None, 5, 22, 48)      │        43,248 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_2 (Dropout)             │ (None, 5, 22, 48)      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_3 (Conv2D)               │ (None, 3, 20, 64)      │        27,712 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_3 (Dropout)             │ (None, 3, 20, 64)      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_4 (Conv2D)               │ (None, 1, 18, 64)      │        36,928 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_4 (Dropout)             │ (None, 1, 18, 64)      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (Flatten)               │ (None, 1152)           │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 100)            │       115,300 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ activation (Activation)         │ (None, 100)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_5 (Dropout)             │ (None, 100)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 50)             │         5,050 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ activation_1 (Activation)       │ (None, 50)             │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_6 (Dropout)             │ (None, 50)             │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (Dense)                 │ (None, 10)             │           510 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ activation_2 (Activation)       │ (None, 10)             │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_7 (Dropout)             │ (None, 10)             │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_3 (Dense)                 │ (None, 2)              │            22 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 252,236 (985.30 KB)
 Trainable params: 252,236 (985.30 KB)
 Non-trainable params: 0 (0.00 B)


```

### Applying the trained neural network



