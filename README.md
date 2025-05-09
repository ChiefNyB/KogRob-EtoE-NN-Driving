# KogRob: Robot control with end-to-end neural network
 
Based on ROS2 Jazzy and Gazebo Harmonic.

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

For utilising, make sure messages are being published on the `/image/compressed` topic (from the camera) and the `/joy_xy` topic (from the `joy_teleop_manual.launch.py` or a similar source).
Open a new terminal and run the node:

```bash
ros2 run KogRob-EtoE-NN-Driving image_recorder
```
You can control the recording by:
*   Press the `r` key in the terminal where `image_recorder` is running to **start** recording.
*   Press `r` again to **stop** recording. You can toggle recording on and off as needed while driving the robot.
*   Press `q` or to **quit** the recorder node gracefully.

When recording is active, the node listens for synchronized image and joystick messages. It will save the compressed image **only if either the X or Y joystick value (or both) is non-zero**. This avoids saving images when the robot is stationary.


The training images are saved under the folder```labelled_data``` with a name similar to this: ```..._Xn0p5_Y0p86.png```

Here the number after ```X``` represents the normalized linear (forward-backward) speed (in this case: -0.5), while the number after ```Y``` denotes the normalized angular (left-right) speed (in this example: 0.86).

The ```test_image_recorder``` node can be used for test and debug purposes only:
```bash
ros2 run KogRob-EtoE-NN-Driving test_image_recorder
```
It works with Twist messages on the /cmd_vel topic, and uses a slightly different image notation.

### Neural network creation and teaching
The origin of the CNN model is NVIDIA's [DAVE-2](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/), with the slight modification of outputting linear and angular speed, instead of giving only the reciprical of the turning radius.
The network has an input of 200x66 pixels and X and Y outputs.

If a correctly labelled training dataset exists in the directory mentioned above, the CNN can be created and trained by navigating to the ```py_scripts``` directory and running the script:

```bash
python3 create_and_train_cnn.py
```
The process can be monitored in the terminal, and in case an error happens it is also printed here.
After a completed training procedure, the test evalution graphs pop-up showing the values of testfitting.

### Applying the trained neural network



## 3. Error handling

If colcon build fails because of the following error:

```bash
ModuleNotFoundError: No module named 'catkin_pkg'
```

Try installing it by running the command:

```bash
pip install catkin_pkg
```