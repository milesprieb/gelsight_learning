import sys, getopt
import numpy as np
import cv2
import math
import os
from os import listdir
from os.path import isfile, join
import open3d
import copy
from gelsight import gsdevice
from gelsight import gs3drecon
#from gelsightcore import poisson_reconstruct
import rospy
from sensor_msgs.msg import PointCloud2
import std_msgs.msg 
import sensor_msgs.point_cloud2 as pcl2


from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def get_diff_img(img1, img2):
    return np.clip((img1.astype(int) - img2.astype(int)), 0, 255).astype(np.uint8)


def get_diff_img_2(img1, img2):
    return (img1 * 1.0 - img2) / 255. + 0.5


def main(argv):

    rospy.init_node('showmini3dros', anonymous=True)

    bridge = CvBridge()

    # number = 9

    device = "mini"
    try:
        opts, args = getopt.getopt(argv, "hd:", ["device="])
    except getopt.GetoptError:
        print('python show3d.py -d <device>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('show3d.py -d <device>')
            print('Use R1 for R1 device, and gsr15???.local for R2 device')
            sys.exit()
        elif opt in ("-d", "--device"):
            device = arg
        elif opt in ("-n", "--number"):
            number = arg

    # Set flags
    SAVE_VIDEO_FLAG = False
    GPU = False
    MASK_MARKERS_FLAG = True
    FIND_ROI = False
    PUBLISH_ROS_PC = True
    SHOW_3D_NOW = True

    # Path to 3d model
    path = '.'

    # Set the camera resolution
    # mmpp = 0.0887  # for 240x320 img size
    mmpp = 0.081  # r2d2 gel 18x24mm at 240x320

    if device == "R1":
        finger = gsdevice.Finger.R1
        # mmpp = 0.1778  # for 160x120 img size from R1
        # mmpp = 0.0446  # for 640x480 img size R1
        # mmpp = 0.029 # for 1032x772 img size from R1
    elif device[-5:] == "local":
        finger = gsdevice.Finger.R15
        capturestream = "http://" + device + ":8080/?action=stream"
    elif device == "mini":
        finger = gsdevice.Finger.MINI
        mmpp = 0.0625
    else:
        print('Unknown device name')
        print('Use R1 for R1 device \ngsr15???.local for R1.5 device \nmini for mini device')

    # This is meters per pixel that is used for ros visualization
    mpp = mmpp / 1000.

    # the device ID can change after chaning the usb ports.
    # on linux run, v4l2-ctl --list-devices, in the terminal to get the device ID for camera
    # if not number:
    number = gsdevice.get_camera_id("GelSight Mini")
    dev = gsdevice.Camera(finger, number)
    net_file_path = '../nnmini.pt'

    dev.connect()

    ''' Load neural network '''
    model_file_path = path
    net_path = os.path.join(model_file_path, net_file_path)
    print('net path = ', net_path)
    gpuorcpu = "cpu"
    if GPU:
        gpuorcpu = "cuda"
    if device == "R1":
        nn = gs3drecon.Reconstruction3D(gs3drecon.Finger.R1, dev)
    else:
        nn = gs3drecon.Reconstruction3D(gs3drecon.Finger.R15, dev)
    net = nn.load_nn(net_path, gpuorcpu)

    if SAVE_VIDEO_FLAG:
        #### Below VideoWriter object will create a frame of above defined The output is stored in 'filename.avi' file.
        file_path = './3dnnlive.mov'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(file_path, fourcc, 60, (160, 120), isColor=True)

    if PUBLISH_ROS_PC:
        ''' ros point cloud initialization '''
        x = np.arange(dev.imgh) * mpp
        y = np.arange(dev.imgw) * mpp
        X, Y = np.meshgrid(x, y)
        points = np.zeros([dev.imgw * dev.imgh, 3])
        points[:, 0] = np.ndarray.flatten(X)
        points[:, 1] = np.ndarray.flatten(Y)
        Z = np.zeros((dev.imgh, dev.imgw))  # initialize points array with zero depth values
        points[:, 2] = np.ndarray.flatten(Z)
        gelpcd = open3d.geometry.PointCloud()
        gelpcd.points = open3d.utility.Vector3dVector(points)
        queue = 0
        gelpcd_pub = rospy.Publisher("/gsmini_pcd", PointCloud2, queue_size=queue)
        geldepth_pub = rospy.Publisher("/gsmini_depth", Image, queue_size=queue)
        gelimg_pub = rospy.Publisher("/gsmini_image", Image, queue_size=queue)


    f0 = dev.get_raw_image()
    roi = (0, 0, f0.shape[1], f0.shape[0])

    print('roi = ', roi)
    print('press q on image to exit')

    ''' use this to plot just the 3d '''
    if SHOW_3D_NOW:
        if device == 'mini':
            vis3d = gs3drecon.Visualize3D(dev.imgh, dev.imgw, '', mmpp)
        else:
            vis3d = gs3drecon.Visualize3D(dev.imgw, dev.imgh, '', mmpp)

    try:
        rate = rospy.Rate(100)
        while not rospy.is_shutdown():

            # get the roi image
            f1 = dev.get_image(roi)
            gelimg_pub.publish(bridge.cv2_to_imgmsg(f1, "bgr8"))
            #bigframe = cv2.resize(f1, (f1.shape[1] * 2, f1.shape[0] * 2))
            #cv2.imshow('Image', bigframe)

            # compute the depth map
            dm = nn.get_depthmap(f1, MASK_MARKERS_FLAG)
            #geldepth_pub.publish(bridge.cv2_to_imgmsg(dm*1000, "passthrough"))

            ''' Display the results '''
            if SHOW_3D_NOW:
                vis3d.update(dm)

        
            #print ('publishing ros point cloud')
            dm_ros = copy.deepcopy(dm) * mpp
            ''' publish point clouds '''
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = 'map'
            points[:, 2] = np.ndarray.flatten(dm_ros)
            gelpcd.points = open3d.utility.Vector3dVector(points)
            gelpcdros = pcl2.create_cloud_xyz32(header, np.asarray(gelpcd.points))
            gelpcd_pub.publish(gelpcdros)

            # -----------------------------------------
            # gelpcd.points: A (x,y,z) point cloud -> (224x224x1) image
            size = (224,224) # Output image size

            y_scalar = 0.0149375 # Max y value
            x_scalar = 0.0199375 # Max x value
            depth_scalar = 0.005 / np.iinfo(np.uint16).max # Max depth value

            depth = np.zeros(size, dtype=np.uint32) 
            depth_count = np.zeros(size, dtype=np.uint16)

            # Bin the points into each pixel, devide by the number of points in each pixel
            for point in gelpcd.points:
                x = int((point[0] / x_scalar)*size[0])
                y = int((point[1] / y_scalar)*size[1])
                z = point[2] / depth_scalar
                #print(x,y,z)
                if not (x >= size[0] or y >= size[1]) and (z<0):
                    depth[x][y] += -z
                    depth_count[x][y] += 1
            depth = np.divide(depth, depth_count, out=depth, where=depth_count!=0, casting='unsafe')

            # Publish the depth image
            depth = depth.astype(np.uint16)
            geldepth_pub.publish(bridge.cv2_to_imgmsg(depth, "passthrough"))
            # -----------------------------------------            

            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break
            if SAVE_VIDEO_FLAG:
                out.write(f1)

            #rate.sleep()

    except KeyboardInterrupt:
        print('Interrupted!')
        dev.stop_video()


if __name__ == "__main__":
    main(sys.argv[1:])
