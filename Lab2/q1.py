import cv2 as cv
# import matplotlib.pyplot as plt
import numpy as np

img=cv.imread('assets/flower.jpg', 0)
data=img.flatten()
hist,_=np.histogram(data,256,[0,256])
cdf=hist.cumsum()
cdf_n=cdf*float(hist.max()) / cdf.max()
new_cdf = np.interp(data, range(256), cdf_n)

new_image = new_cdf.reshape(img.shape).astype(np.uint8)
# plt.plot(cdf_n)
# plt.hist(data,256,[0,256])
# plt.show()
cv.imwrite('assets/hist_eq.jpg', new_image)
cv.imshow('image',new_image)
cv.waitKey(0)