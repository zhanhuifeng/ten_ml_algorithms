"""
1. K-MEANS CLUSTERING
    K-means is a popular unsupervised machine-learning algorithm that is used for clustering data.
    It works by dividing a set of data points into a specified number of clusters, where each data
    point belongs to the cluster with the nearest mean. K-means is an iterative algorithm that
    repeats the clustering process until convergence is achieved.
    The k-means algorithm is easier to train compared to other clustering algorithms. It is scalable
    on large datasets for clustering samples. It is simple to implement and interpret. Let's look at
    some use cases of the k-means algorithm.
2. USE-CASES:
    2.1. Customer segmentation
    2.2. Anomaly detection
    2.3. Medical image segmentation

3. Example:
    Let's look at the code implementation of the K-Means Clustering algorithm.

"""
import numpy as np
from sklearn.cluster import KMeans

# Sample data
x = np.array([[1, 2], [1, 3], [2, 2], [2, 3], [5, 6], [6, 5], [7, 7], [8, 6]])

# Create and fit the K-Means model with k=2 (2 clusters)

kmeans = KMeans(n_clusters=2)
kmeans.fit(x)

# Get cluster centroids and labels
centroids = kmeans.cluster_centers_
print("centroids", centroids)
labels = kmeans.labels_
print("labels", labels)

# Predict cluster membership for new data points
new_data = np.array([[3, 4], [6, 6]])
new_labels = kmeans.predict(new_data)
print("new_labels", new_labels)
