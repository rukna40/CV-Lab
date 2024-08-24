import cv2 as cv

img = cv.imread('/Lab1/assets/photo-8434386_1280.jpg')
print("Image Shape = ", img.shape)
x1,y1,x2,y2 = input("Enter coordinates of rectangle: ").split()
img=cv.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), (255,0,0), 2)
cv.imshow('img',img)
cv.waitKey(0)
# cv.destroyAllWindows()