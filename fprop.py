import numpy as np
# Sample data
X = [[1, 2, 3]]

# Define the parameters of the network
W1 = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
b1 = [[0.1], [0.2]]
W2 = [[0.7, 0.8]]
b2 = [[0.3]]

# Perform forward propagation
Z1 = [[sum(x * w for x, w in zip(X[0], W1_row)) + b1_row[0] for W1_row, b1_row in zip(W1, b1)]]
A1 = [[max(0, z) for z in Z1_row] for Z1_row in Z1]  # ReLU activation
Z2 = [[sum(a * w for a, w in zip(A1[0], W2_row)) + b2_row[0] for W2_row, b2_row in zip(W2, b2)]]
A2 = [[1 / (1 + pow(2.71828, -z)) for z in Z2_row] for Z2_row in Z2]  # Sigmoid activation

# Print the final output
print(A2)


import numpy as np

# Sample data
X = np.array([[1, 2, 3]])

# Define the parameters of the network
W1 = np.random.randn(2, 3)
b1 = np.random.randn(2, 1)
W2 = np.random.randn(1, 2)
b2 = np.random.randn(1, 1)

# Perform forward propagation
Z1 = np.dot(W1, X.T) + b1
A1 = np.maximum(0, Z1)  # ReLU activation
Z2 = np.dot(W2, A1) + b2
A2 = 1 / (1 + np.exp(-Z2))  # Sigmoid activation

# Print the final output
print(A2)

