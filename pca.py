import numpy as np

# Sample data
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

# Standardize the data (optional but recommended)
X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Compute the covariance matrix
covariance = np.cov(X_std, rowvar=False)

# Compute the eigenvectors and eigenvalues
eigenvalues, eigenvectors = np.linalg.eig(covariance)

# Sort the eigenvalues in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

# Select the top k eigenvectors
k = 2
selected_eigenvectors = sorted_eigenvectors[:, :k]

# Project the data onto the selected eigenvectors
transformed_data = np.dot(X_std, selected_eigenvectors)

# Print the transformed data
print("Transformed data:")
print(transformed_data)

from sklearn.decomposition import PCA
import numpy as np

# Sample data
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

# Create a PCA object with the desired number of components
pca = PCA(n_components=2)

# Fit the PCA model to the data
pca.fit(X)

# Transform the data to the lower-dimensional space
transformed_data = pca.transform(X)

# Print the transformed data
print("Transformed data:")
print(transformed_data)

