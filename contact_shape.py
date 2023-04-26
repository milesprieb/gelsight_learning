import cv2 
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('/home/rpmdt05/Code/Tacto_good/data_aug/data_mod/Blur_Depth_bishop21.jpg', cv2.IMREAD_UNCHANGED)
print(img/256)

contact = np.where(img/256 > 0, 1.0, 0.0)
print(contact)
fig = plt.figure(figsize=(10, 10))
fig.add_subplot(1, 2, 1)
plt.imshow(img)
fig.add_subplot(1, 2, 2)
plt.imshow(contact )
plt.show()