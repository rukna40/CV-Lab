import cv2
import numpy as np

img=cv2.imread('/home/ankur/Codes/Sem5/CV-Lab/Lab5/square.jpg',0)

dx=cv2.Sobel(src=img,dx=1,dy=0,ddepth=-1)
dy=cv2.Sobel(src=img,dx=0,dy=1,ddepth=-1)


A=cv2.GaussianBlur(dx**2,(3,3),0)
B=cv2.GaussianBlur(dy**2,(3,3),0)
C=cv2.GaussianBlur(dx*dy,(3,3),0)

k=0.05
hcg=(A*B - (C*C)) - k*(A + B)*(A + B)

cv2.imshow('Original',img)
cv2.imshow('hcg',hcg)

cv2.waitKey(0)
cv2.destroyAllWindows()