import cv2 as cv

img = cv.imread('/Lab1/assets/photo-8434386_1280.jpg')
print("Image Shape = ", img.shape)
x,y=input("Enter pixel coordinates: ").split()
b,g,r=img[int(y)][int(x)]
print("R = ",r,"\nG = ",g,"\nB = ",b)
