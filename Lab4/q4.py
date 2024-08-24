import cv2
import numpy as np

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def assign_clusters(data, centroids):
    clusters = np.zeros(len(data), dtype=int)
    for i in range(len(data)):
        distances = [euclidean_distance(data[i], centroid) for centroid in centroids]
        clusters[i] = np.argmin(distances)
    return clusters

def update_centroids(data, clusters, k):
    new_centroids = np.zeros((k, data.shape[1]))
    for i in range(k):
        points_in_cluster = data[clusters == i]
        if len(points_in_cluster) > 0:
            new_centroids[i] = np.mean(points_in_cluster, axis=0)
    return new_centroids

def has_converged(old_centroids, new_centroids, tolerance=1e-6):
    return np.all(np.linalg.norm(new_centroids - old_centroids, axis=1) < tolerance)

def k_means(data, k, max_iters=100, tolerance=1e-6):
    np.random.seed(0)
    initial_indices = np.random.choice(len(data), k, replace=False)
    centroids = data[initial_indices]

    for i in range(max_iters):
        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, clusters, k)
        if has_converged(centroids, new_centroids, tolerance):
            print(f"Converged after {i+1} iterations")
            break
        centroids = new_centroids

    return centroids, clusters

# Read and process the image
img = cv2.imread('../assets/shapes.png')
if img is None:
    raise ValueError("Image not found or unable to load.")

# Convert the image to float32 for k-means clustering
data = img.reshape(-1, 3).astype(np.float32)

# Apply k-means clustering
k = 6
centroids, clusters = k_means(data, k)

# Reconstruct the clustered image
reconstructed_image = centroids[clusters].reshape(img.shape).astype(np.uint8)

# Display the images
cv2.imshow('Original Image', img)
cv2.imshow('Clustered Image', reconstructed_image)

# Wait until a key is pressed and then close the image windows
cv2.waitKey(0)
cv2.destroyAllWindows()
