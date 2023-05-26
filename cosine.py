import numpy as np

# Example input data
A = np.array([1, 2, 3, 4, 5])
B = np.array([2, 4, 6, 8, 10])

# Compute cosine similarity
dot_product = np.dot(A, B)
magnitude_A = np.linalg.norm(A)
magnitude_B = np.linalg.norm(B)
similarity = dot_product / (magnitude_A * magnitude_B)

print(similarity)

from sklearn.metrics.pairwise import cosine_similarity

# Example input data
A = [[1, 2, 3, 4, 5]]
B = [[2, 4, 6, 8, 10]]

# Compute cosine similarity
similarity = cosine_similarity(A, B)

print(similarity)

