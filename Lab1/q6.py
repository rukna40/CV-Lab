import cv2 as cv

img = cv.imread('/Lab1/assets/photo-8434386_1280.jpg')
img=cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
cv.imshow('Image', img)
cv.waitKey(0)
# cv.destroyAllWindows()