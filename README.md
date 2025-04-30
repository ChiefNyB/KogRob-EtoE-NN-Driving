# KogRob: Robot control with end-to-end neural network
 
Based on ROS2 Jazzy and Gazebo Harmonic.

Dependencies:

- MOGI turtlebot3 repository

You can add these with cloning the following repos into the src folder:

```bash
git clone -b mogi-ros2 https://github.com/MOGI-ROS/turtlebot3
```
Install the following packages using apt:
```bash
sudo apt install ros-jazzy-joy
sudo apt install ros-jazzy-teleop-twist-joy
```
Don't forget to source the setup:

```bash
source ~/ros2_project/install/setup.bash
```

If colcon build fails because of the following error:

```bash
ModuleNotFoundError: No module named 'catkin_pkg'
```

Try installing it by running the command:

```bash
pip install catkin_pkg
```

# Control
Manual control with joystick using the launch file:
```bash
ros2 launch KogRob-EtoE-NN-Driving joy_teleop_manual.launch.py
```
You can configure the joystick teleop in config/teleop_joy.yaml