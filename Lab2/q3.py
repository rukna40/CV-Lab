import cv2 as cv

img = cv.imread('assets/tiger.jpg')
img=cv.resize(img,(500,500))

crop_img = img[80:280, 150:330]
cv.imshow('Image', img)
cv.imshow('Crop', crop_img)

cv.waitKey(0)
cv.destroyAllWindows()