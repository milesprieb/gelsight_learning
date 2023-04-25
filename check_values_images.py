import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def check_values(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    print(img.shape)
    print(np.max(img))
    print(np.min(img))
    plt.imshow(img)
    plt.show()


def main():
    path = 'data_mod/Blur_Depth_bishop21.jpg'
    check_values(path)

if __name__ == '__main__':
    main()