import cv2
import numpy as np
import matplotlib.pyplot as plt

def cornerTest(center, pixels, threshold):
    brighter=np.sum(pixels > center + threshold)
    darker=np.sum(pixels < center - threshold)
    return brighter >= 9 or darker >= 9

def findCorners(image, threshold):
    rows, cols = image.shape
    corners = []

    for x in range(3, rows - 3):
        for y in range(3, cols - 3):
            center = image[x, y]
            pixels= [
                image[x - 3, y], image[x - 3, y + 1], image[x - 2, y + 2],
                image[x - 1, y + 3], image[x, y + 3], image[x + 1, y + 3],
                image[x + 2, y + 2], image[x + 3, y + 1], image[x + 3, y],
                image[x + 3, y - 1], image[x + 2, y - 2], image[x + 1, y - 3],
                image[x, y - 3], image[x - 1, y - 3], image[x - 2, y - 2],
                image[x - 3, y - 1]
            ]

            if cornerTest(center, pixels, threshold):
                corners.append((x, y))

    return corners

def visualize_corners(image, corners):
    image_with_corners = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for corner in corners:
        cv2.circle(image_with_corners, (corner[1], corner[0]), 3, (0, 255, 0), -1)

    plt.imshow(image_with_corners)
    plt.show()

image = cv2.imread('/home/ankur/Codes/Sem5/CV-Lab/Lab5/house.png', 0)
corners=findCorners(image, 20)
visualize_corners(image, corners)

