import cv2
import numpy as np

def hough_transform(image, rho_res=1, theta_res=np.pi / 180, threshold=100):
    edges = cv2.Canny(image, 50, 150, apertureSize=3)

    height, width = edges.shape
    diag_len = int(np.sqrt(height ** 2 + width ** 2))
    rhos = np.arange(-diag_len, diag_len, rho_res)
    thetas = np.arange(-np.pi, np.pi, theta_res)

    accumulator = np.zeros((len(rhos), len(thetas)), dtype=int)

    y_idxs, x_idxs = np.nonzero(edges)

    for x, y in zip(x_idxs, y_idxs):
        for theta_idx in range(len(thetas)):
            theta = thetas[theta_idx]
            rho = int(x * np.cos(theta) + y * np.sin(theta))
            rho_idx = np.argmin(np.abs(rhos - rho))
            accumulator[rho_idx, theta_idx] += 1

    lines = []
    for rho_idx, theta_idx in np.argwhere(accumulator > threshold):
        rho = rhos[rho_idx]
        theta = thetas[theta_idx]
        lines.append((rho, theta))

    return lines, edges, accumulator

def draw_lines(image, lines):
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    height, width = image.shape
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(output, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return output

image = cv2.imread('../assets/road.png',0)

lines, edges, accumulator = hough_transform(image, rho_res=1, theta_res=np.pi / 180, threshold=100)

result_image = draw_lines(image, lines)

cv2.imshow('Original Image', image)
cv2.imshow('Edges', edges)
cv2.imshow('Detected Lines', result_image)

cv2.waitKey(0)
cv2.destroyAllWindows()



