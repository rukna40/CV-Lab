
# #Implementation of SIFT descriptor using Opencv library 
import cv2
import matplotlib.pyplot as plt
from cv2 import SIFT_create
import numpy as np

# read images
img = cv2.imread(r'/home/ankur/Codes/Sem5/CV-Lab/Lab4/colors.png')
img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# create sift object
sift = SIFT_create()
keypoints, descriptors =sift.detectAndCompute(img, None)
keypoints_with_size = np.copy(img)
cv2.drawKeypoints (img, keypoints, keypoints_with_size, color = (255, 0, 0), flags= cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(keypoints_with_size),
plt.show()