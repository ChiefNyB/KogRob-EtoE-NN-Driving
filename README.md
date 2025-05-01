# KogRob: Robot control with end-to-end neural network
 
Based on ROS2 Jazzy and Gazebo Harmonic.

## 1. Install and setup

Dependencies (beyond ROS2 and Gazebo):
- MOGI turtlebot3 repository
- Python3
- Some python packages (installed automaically, see later)
- ROS2 joystick interface package

You can add these with cloning the following repo into the src folder:

```bash
git clone -b mogi-ros2 https://github.com/MOGI-ROS/turtlebot3
```
And installing the packages below using apt:
```bash
sudo apt install python3-pip
sudo apt install pipx
sudo apt install ros-jazzy-joy
sudo apt install ros-jazzy-teleop-twist-joy
```

Set up a Python virtual enviroment, as described [here](https://github.com/MOGI-ROS/Week-1-8-Cognitive-robotics?tab=readme-ov-file#line-following). 
If not already installed, the ```colcon build``` will install the python dependencies needed to run the package content.
Because the CNN training is a standalone python script (not a ROS2 node), it is necessary to make it executable, by navigating to the ```py_scripts``` directory within the packge and running the following command:
```bash
chmod +x create_and_train_cnn.py
```

Also don't forget to source the setup (workspace path may be different):

```bash
source ~/ros2_project/install/setup.bash
```

For GPU acceleratied training process CUDA and cuDNN Library are required (optional).


## 2. How to use

### Controller
Manual control with joystick can be started using the launch file:
```bash
ros2 launch KogRob-EtoE-NN-Driving joy_teleop_manual.launch.py
```
You can configure the joystick teleop in config/teleop_joy.yaml

### Labelled data acquisition
...

Finally, the training images are saved under the ../labelled_data folder.

### Neural network creation and teaching
If a correctly labelled training dataset exists in the directory mentioned above, the CNN can be created and trained, by navigating to the ```py_scripts``` directory and running the following command:

```bash
python3 create_and_train_cnn.py
```
The process can be followed in the terminal, and in case an error happens it is also printed here.

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




