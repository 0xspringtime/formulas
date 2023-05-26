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

# Calculate the WCSS
wcss = np.sum((X - centroids[labels])**2)

# Print the WCSS
print("WCSS:", wcss)

from sklearn.cluster import KMeans
import numpy as np

# Sample data
X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])

# Create a K-means clustering model
kmeans = KMeans(n_clusters=2)

# Fit the model to the data
kmeans.fit(X)

# Calculate the WCSS
wcss = kmeans.inertia_

# Print the WCSS
print("WCSS:", wcss)

