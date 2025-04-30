# KogRob: Robot control with end-to-end neural network
 
Based on ROS2 Jazzy and Gazebo Harmonic.

## 1. Install and setup:

Dependencies:
- MOGI turtlebot3 repository
- Python3

You can add these with cloning the following repos into the src folder:

```bash
git clone -b mogi-ros2 https://github.com/MOGI-ROS/turtlebot3
```
Install the following packages using apt:
```bash
sudo apt install python3-pip
sudo apt install pipx
```

Set up a Python virtual enviroment, as described [here](https://github.com/MOGI-ROS/Week-1-8-Cognitive-robotics?tab=readme-ov-file#line-following)
If not already installed, the colcon build will install the python dependencies needed to run the package content.

Don't forget to source the setup:

```bash
source ~/ros2_project/install/setup.bash
```

For GPU acceleration CUDA and cuDNN Library are required.



## 2. Error handling

If colcon build fails because of the following error:

```bash
ModuleNotFoundError: No module named 'catkin_pkg'
```

Try installing it by running the command:

```bash
pip install catkin_pkg
```




