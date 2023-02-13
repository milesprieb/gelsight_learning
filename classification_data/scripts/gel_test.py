import cv2 as cv
#import gelsight as gs
import numpy as np
from datetime import datetime

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
    # Capture frame-by-frame
    ret, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    #rint(frame.shape[0],frame.shape[1])
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    # Display the resulting frame
    
    cv.imshow('frame1', frame1)
    cv.imshow('frame2', frame2)
    
    if cv.waitKey(1) == ord('q'):
        cv.imwrite('left_knight_{}.jpg'.format(datetime.now()),frame1)
        cv.imwrite('right_knight_{}.jpg'.format(datetime.now()),frame2)

    if cv.waitKey(1) == ord('k'):
        break

# When everything done, release the capture
cap1.release()
cap2.release()
cv.destroyAllWindows()
    