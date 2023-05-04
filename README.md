# Modules
## [Gelsight Software](https://github.com/gelsightinc/gsrobotics)
This is the main software package for the GelSight tactile sensor. It contains the ROS interfaces for the sensor, as well as the calibration and visualization tools. 

- Clone the repository and install the dependencies:
```
pip3 install .
PYDIR=`pip3 show gelsight | grep -i location | cut -f2 -d" "`
export PYTHONPATH=$PYDIR/gelsightcore:$PYDIR/gelsight:$PYTHONPATH
sudo apt-get update
sudo apt-get -y install v4l-utils
```
- Write ```./src/show3d_ros.py``` to ```/examples/ros/show3d_ros.py```
- Run with ```python3 show3d_ros.py``` with the ros master running.

This should publish the point cloud to the topic ```/gsmini_pcd```. 

It also publishes the ```(224,224x1)``` depth image to the topic ```/gsmini_depth```. 

The origional image is published to the topic ```/gsmini_image```. 

**Note:** The depth image is not the same as the color image from the sensor. It is the depth image from the point cloud.

**Note:** The ID of the GelSight camera needs to be set manually. This is done in the ```show3d_ros.py``` file. Guess and check until the correct ID is found.


## [Tacto Simulator](https://github.com/facebookresearch/tacto)
This is the simulator for the GelSight tactile sensor. It contains the simulation environment for the sensor. This takes a 3D mesh as input and simulates the sensor output. 

 Clone the repository and install the dependencies:
```
pip install tacto
pip install -r requirements/examples.txt
```

- Write ```./src/demo_pybullet_digit.py``` to ```/examples/demo_pybullet_digit.py```
- Copy ```./src/objects.zip``` to ```/examples/``` and extract. 
- Run with ```python3 demo_pybullet_digit.py``` to start rendering.

**Note:** The ```demo_pybullet_digit.py``` file is a modified version of the origional ```demo_pybullet.py``` file. It is modified to work with the ```objects.zip``` file.

**Note:** The ```objects.zip``` file contains the 3D mesh of the chess pieces used in the simulation. It also contains the ```urdf``` files for the chess pieces.

