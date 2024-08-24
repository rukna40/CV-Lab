import cv2 as cv
import numpy as np

image = cv.imread('assets/flower.jpg')

size=5
sigma = 1.0

kernel1 = np.zeros((size, size))
center = size // 2

for x in range(size):
    for y in range(size):
        x_dist = x - center
        y_dist = y - center
        kernel1[x, y] = (1 / (2 * np.pi * sigma ** 2)) * np.exp(- (x_dist ** 2 + y_dist ** 2) / (2 * sigma ** 2))

kernel1 = kernel1 / np.sum(kernel1)
gaussian_img=cv.filter2D(image,-1,kernel1)

kernel2=np.ones((5,5),np.float32)/25
box_img=cv.filter2D(image,-1,kernel2)

cv.imshow('img',image)
cv.imshow('gaussian_img',gaussian_img)
cv.imshow('box_img',box_img)

cv.waitKey(0)
cv.destroyAllWindows()
