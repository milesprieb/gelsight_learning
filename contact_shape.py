import cv2 
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('real_depth/gs_data/Depth_knight_1682108172_261467.png', cv2.IMREAD_UNCHANGED)
print(img)
plt.imshow(img)
plt.show()
contact = np.where(img > 0, 1, 0)
print(contact)
plt.imshow(contact)
plt.show()