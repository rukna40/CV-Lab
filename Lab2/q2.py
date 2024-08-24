import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img1=cv.imread('assets/flower.jpg', 0)
img2=cv.imread('assets/tiger.jpg', 0)

data1=img1.flatten()
data2=img2.flatten()

hist,_=np.histogram(data1,256,[0,256])
cdf=hist.cumsum()
cdf_n=cdf*float(hist.max()) / cdf.max()

new_cdf = np.interp(data2, range(256), cdf_n)

new_image = new_cdf.reshape(img2.shape).astype(np.uint8)

cv.imwrite('assets/hist_eq.jpg', new_image)
cv.imshow('image',new_image)
cv.waitKey(0)