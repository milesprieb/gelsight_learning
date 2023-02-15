import cv2 as cv
#import gelsight as gs
import numpy as np
from datetime import datetime
import argparse
from pathlib import Path
import os

parser = argparse.ArgumentParser(description='Extract gelsight data from UR5 arm')
parser.add_argument('--path', type=Path, default=0, help='Camera 1')
args = parser.parse_args()
cap1 = cv.VideoCapture(0)
cap2 = cv.VideoCapture(2)

cv.namedWindow('frame1', cv.WINDOW_NORMAL)
cv.resizeWindow('frame1', height=300,width=700)
cv.namedWindow('frame2', cv.WINDOW_NORMAL)
cv.resizeWindow('frame2', height=300,width=700)
file_path = './3dlive_{}.mov'.format(datetime.now())

if not cap1.isOpened():
    print("Cannot open camera")
    exit()
while True:
    if  not args.path.exists():
        print('Invalid save path')
        break
    # Capture frame-by-frame
    ret, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    #rint(frame.shape[0],frame.shape[1])
    # if frame is read correctly ret irint(args.path, type(args.path),s True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
        
    # print(os.path.join(args.path, 'left_knight_{}.jpg'.format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))))
    # Our operations on the frame come here
    # Display the resulting frame
    
    cv.imshow('frame1', frame1) #This is for left wrt the arms
    cv.imshow('frame2', frame2) #This is for right wrt the arms
    
    if cv.waitKey(1) == ord('q'):
        print(os.path.join(args.path, '****_knight_{}.jpg'.format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))))
        cv.imwrite(os.path.join(args.path, 'left_knight_{}.jpg'.format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))),frame1)
        cv.imwrite(os.path.join(args.path, 'right_knight_{}.jpg'.format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))),frame2)

    if cv.waitKey(1) == ord('k'):
        break

# When everything done, release the capture
cap1.release()
cap2.release()
cv.destroyAllWindows()
    