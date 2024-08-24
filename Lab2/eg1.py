import cv2
import numpy as np


img = cv2.imread('assets/flower.jpg')


c = 255 / (np.log(1 + np.max(img)))

log_transformed = c * np.log(1 + img)

log_transformed = np.array(log_transformed, np.uint8)

cv2.imwrite('images/log_transformed.jpg', log_transformed)