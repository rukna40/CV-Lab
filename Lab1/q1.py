import cv2 as cv

img = cv.imread('/Lab1/assets/photo-8434386_1280.jpg')
cv.imshow('Image', img)
cv.waitKey(0)
# cv.destroyAllWindows()
cv.imwrite('/Lab1/assets/output.jpg', img)
