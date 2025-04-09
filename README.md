# KogRob: Robot control with end-to-end neural network
 
Based on ROS2 Jazzy and Gazebo Harmonic.

Dependencies:

- MOGI turtlebot3 repository

You can add these with cloning the following repos into the src folder:

```bash
git clone -b mogi-ros2 https://github.com/MOGI-ROS/turtlebot3
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