import cv2
import numpy as np

img = cv2.imread('zg',0)
thresh=255/2
binary = np.where(img > thresh, 255, 0).astype(np.uint8)
cv2.imshow('binary', binary)

if cv2.waitKey(0) & 0xff == ord('q'):
    cv2.destroyAllWindows()
