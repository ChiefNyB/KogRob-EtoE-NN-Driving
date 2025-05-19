from setuptools import find_packages, setup
import os # Import os
from glob import glob # Import glob

package_name = 'KogRob-EtoE-NN-Driving'
submodule_name = 'py_scripts' # Define the directory containing the script

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']), # Should find 'py_scripts' if it has __init__.py or is a namespace package
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Example: Include launch files if you have them
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'gazebo_models', 'light_bg'),
            [
                'gazebo_models/light_bg/model.config',
                'gazebo_models/light_bg/model.sdf',
            ]
        ),
        (os.path.join('share', package_name, 'gazebo_models', 'light_bg', 'meshes'),
            [
                'gazebo_models/light_bg/meshes/light_bg.dae',
            ]
        )
    ],
    # Python requires
    install_requires=[
        'setuptools',
        'tensorflow==2.18.0',
        'scikit-learn',
        'imutils',
        'numpy==1.26.4',
        'opencv-python',
        'matplotlib',
        ],
    
    zip_safe=True,
    maintainer='barney',
    maintainer_email='105289927+ChiefNyB@users.noreply.github.com',
    description='Package for training and running an End-to-End CNN for Turtlebot3 driving.',
    license='Apache 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "image_recorder = py_scripts.image_recorder:main",
            "joy_republisher = py_scripts.joy_republisher:main",
            "joy_xy_republisher = py_scripts.joy_xy_republisher:main",
            "joy_xy_publisher = py_scripts.joy_xy_publisher:main",
            "joy_cnn_drive = py_scripts.joy_cnn_drive:main",
        ],
    },
)
