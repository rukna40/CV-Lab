import cv2 as cv

img = cv.imread('/Lab1/assets/photo-8434386_1280.jpg')
img=cv.resize(img,(500,500))
cv.imshow('Image', img)
cv.waitKey(0)
# cv.destroyAllWindows()
