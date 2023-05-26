import numpy as np

# Sample data
X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])

# Number of clusters
K = 2

# Initialize cluster centroids randomly
centroids = X[np.random.choice(X.shape[0], K, replace=False)]

# Assign data points to clusters
distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
labels = np.argmin(distances, axis=0)

# Update centroids
new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(K)])

# Repeat until convergence
while not np.allclose(centroids, new_centroids):
    centroids = new_centroids
    distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
    labels = np.argmin(distances, axis=0)
    new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(K)])

# Print the cluster labels and centroids
print("Cluster labels:", labels)
print("Centroids:", new_centroids)

from sklearn.cluster import KMeans
import numpy as np

# Sample data
X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])

# Create a K-means clustering model
kmeans = KMeans(n_clusters=2)

# Fit the model to the data
kmeans.fit(X)

# Get the cluster labels and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Print the cluster labels and centroids
print("Cluster labels:", labels)
print("Centroids:", centroids)

