import cv2
import numpy as np

image_path = '../assets/shapes.png'
image = cv2.imread(image_path)

hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_bound = np.array([30, 50, 50])
upper_bound = np.array([90, 255, 255])

mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
segmented_image = cv2.bitwise_and(image, image, mask=mask)

cv2.imshow('Original Image', image)
cv2.imshow('Mask', mask)
cv2.imshow('Segmented Image', segmented_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

