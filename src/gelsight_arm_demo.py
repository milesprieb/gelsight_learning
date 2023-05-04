import rospy
import sys
import os
import transforms3d
import random
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common import move_group_interface
import std_msgs.msg
from sensor_msgs.msg import PointCloud2
import ros_numpy
import numpy as np
from tf.transformations import quaternion_from_euler
from sensor_msgs.msg import Image
import cv2
import time
import torch
from classification_blue_model import GelsightResNet
from torchvision.models import resnet
import torch.nn as nn


label_map = {'bishop': 0, 
             'king': 1,
             'knight': 2,
             'pawn': 3,
             'queen': 4,
             'rook': 5,
             }

label_map_inv = {v: k for k, v in label_map.items()}

# Callback for the pointcloud
def gs_force_feedback(data, interface):
    #rospy.loginfo('Force feedback')
    global grip # Flag for pressure threshold
    global grip_move # Flag for closing the gripper
    global grip_pos # Position of the gripper

    global save # File name to save the image (debug)

    if grip:
        data_cloud = ros_numpy.numpify(data)
        force = np.quantile(data_cloud['z'], 0.95)*1e3*6 # Get scaled force pointcloud
        
        # if force > 0.2:
        #     save = "./Rook/Depth_rook_" + str(int(time.time())) + "_" + str(random.randint(0, 1000000)) + ".png"
        # return

        if grip_move: # True implies the gripper has been touched
            if force < 0.2 and grip_move: # Check if below threshold
                grip_pos += 1 # Close a little
            else:  # Done closing
                grip_move = False # Reset the flag
                grip = False # Stop closing the gripper

        else: # False implies the gripper has not been touched
            if force > 0.025:   # Check for small force
                grip_move = True # Start Closing
        
    interface.keep_closing(grip, grip_pos) # Move the gripper to grip_pos


# Callback for the depth image
def img_callback(data):
    #rospy.loginfo('IMG: %s', np.max(ros_numpy.numpify(data)))

    global save # Flag to run NN

    global model # Model for classification
    global device # Device to run the model on
    global piece_type # Type of piece detected

    if save != False: # Check if the flag is set
        data = ros_numpy.numpify(data)
        rospy.loginfo('IMAGE SAVED')
        cv2.imwrite(save, data) 

        depth_image = data/ 65535.0 
        depth_image = depth_image / (depth_image.max() - depth_image.min()) 

        inputs = torch.tensor(depth_image, dtype=torch.float).to(device) # Convert to tensor
        inputs = inputs.unsqueeze(0) # Add channel dimension
        inputs = inputs.unsqueeze(0) # Add batch dimension

        outputs = model(inputs) # Run the model
        _, preds = torch.max(outputs, 1) # Get the prediction
        piece_type = preds.item() # Set Global variable piece_type, this is read in main

        save = False # Reset the flag


def main():
    global grip
    global grip_move
    global grip_pos
    global save

    global model
    global device
    global piece_type

    open_pos = 180

    grip = False
    grip_move = False
    grip_pos = open_pos
    save = False

    # Load the model
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('Device: %s', device)
    model = GelsightResNet(block=resnet.Bottleneck, layers=[3, 4, 6, 3])
    model_ftrs = model.fc.in_features
    model.fc = nn.Linear(model_ftrs, 6)
    model.load_state_dict(torch.load('depth_classifier_430.pth', map_location=torch.device('cpu')))
    model = model.to(device)
    model.eval()

    # Initialize ROS node
    interface = move_group_interface.MoveGroupInterface(single=True)
    rospy.Subscriber("/gsmini_pcd", PointCloud2, gs_force_feedback, interface)
    rospy.Subscriber("/gsmini_depth", Image, img_callback) 
    
    # Set the home and chess positions
    rot = quaternion_from_euler(0, 0, 0, 'rxyz')
    j_home = np.deg2rad(np.array([-88.0, -114.0, 88.0, -247.0, -90.0, -90.0]))
    j_chess = np.deg2rad(np.array([-50.0, -113.0, 115.0, -177.0, -38.0, -185.0]))
    

    while True:
        # grip = True
        # while(grip):
        #     rospy.sleep(0.1)
        # continue

        interface.go_to_joint_state(j_home, interface.mg_lightning) # Go to home position

        # Grip the piece and run the NN
        grip = True
        while(grip):
            rospy.sleep(0.1)
        save = './gs_data/' + str(time.time()) + '.png'
        while(save != False):
            rospy.sleep(0.1)
        rospy.loginfo('Type: %s', label_map_inv[piece_type])
        
        interface.go_to_joint_state(j_chess, interface.mg_lightning) # Go to chess position

        rospy.sleep(2.0)

        grip_pos = open_pos

        rospy.sleep(2.0)


if __name__ == "__main__":
    main()