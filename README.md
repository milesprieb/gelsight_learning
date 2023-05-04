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
- Write [```./src/show3d_ros.py```](src/show3d_ros.py) to ```/examples/ros/show3d_ros.py```
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

- Write [```./src/demo_pybullet_digit.py```](src/demo_pybullet_digit.py) to ```/examples/demo_pybullet_digit.py```
- Copy [```./src/objects.zip```](src/objects.zip) to ```/examples/``` and extract. 
- Run with ```python3 demo_pybullet_digit.py``` to start rendering.

**Note:** The ```demo_pybullet_digit.py``` file is a modified version of the origional ```demo_pybullet.py``` file. It is modified to work with the ```objects.zip``` file.

**Note:** The ```objects.zip``` file contains the 3D mesh of the chess pieces used in the simulation. It also contains the ```urdf``` files for the chess pieces.

## Zeus
This contains the kinematics for the Zeus robot. It also contains the ROS interfaces for the robot. This code comes from somewhere and it is not clear where. The file [```gelsight_arm_demo.py```](src/gelsight_arm_demo.py) is the main file for the robot. It contains the code for the robot to move to the handover position and then grasp with the GelSight sensor. 

The callback function for the pointcloud checks for the pressure on the sensor and uses that to determine when to grasp and when to stop grasping.

The script then moves the piece to the home position and releases the piece.

**Note** This requires the rest of the Zeus code to run. The entire codebase is not included here.


## [Classification Model](https://drive.google.com/drive/folders/1mykcNLeSqjIRARKxkEUXOAmWEU-rAeOe?usp=share_link)
This contains the classification model for the GelSight sensor. It is a pretrained model that can be used to classify the images from the sensor. To use this model, download the folder and place it in the ```/src/``` directory.

- (```classification_blur_model.py```)[/src/classification_blur_model.py] This is the main file for the classification model. The model takes ```224x224x1``` real or blurred simulated image as input and outputs a ```6``` vector. The output is a one-hot encoded vector for the class of the piece. 

This uses a modified ResNet-50 model pretrained on the ImageNet dataset. The model is then fine-tuned on the GelSight dataset.

**Note** The model is trained on the blurred images from the simulation. 

**Note** (```./src/classification_depth_model```)(/src/classification_depth_model) This is the same as the ```classification_blur_model.py``` file, but an earlier version. It is not used in the final code.

 - (```classification_blur_model.py```)[/src/classification_blur_model.py] This is the main file for the classification model. The model takes ```224x224x3``` real or simulated RGB input and outputs a ```6``` vector. The output is a one-hot encoded vector for the class of the piece. 

 **Note** This takes the raw RGB image as input. It does not use the depth image.