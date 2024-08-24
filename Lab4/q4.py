import cv2
import numpy as np

def distance(x,y):
    return np.sqrt(np.sum((x - y) ** 2,axis=1))

def initialize(data,K):
    indices = np.random.choice(len(data), K, replace=False)
    return data[indices]

def compute_clusters(data,centroids):
    clusters = {i: [] for i in range(len(centroids))}
    labels=[]
    for point in data:
        id=np.argmin(distance(point,centroids))
        clusters[id].append(point)
        labels.append(id)
    return clusters,np.array(labels)

def update_centroids(clusters,K,dim):
    centroids=np.zeros((K, dim))
    for id in range(K):
        if len(clusters[id])>0:
            centroids[id]=np.mean(clusters[id],axis=0)       
    return centroids

def kmeans(K,epoch,data):
    dim=data.shape[1]
    centroids=initialize(data,K)
    for iter in range(epoch):
        clusters,labels=compute_clusters(data,centroids)
        centroids=update_centroids(clusters,K,dim)
    return labels,centroids

if __name__=='__main__':
    img = cv2.imread('Lab4/colors.png')
    data = img.reshape(-1, 3)

    K=5
    epoch=20
    labels,centroids=kmeans(K,epoch,data)

    segmented_img = centroids[labels].reshape(img.shape).astype(np.uint8)

    cv2.imshow('Original Image', img)
    cv2.imshow('Segmented Image', segmented_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

