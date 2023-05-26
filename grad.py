import numpy as np

# Sample data
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([3, 7, 11])

# Number of training examples
m = len(y)

# Add a column of ones to X for the intercept term
X = np.c_[np.ones((m, 1)), X]

# Initialize parameters
theta = np.zeros(X.shape[1])

# Hyperparameters
alpha = 0.01
num_iterations = 1000

# Perform gradient descent
for _ in range(num_iterations):
    predictions = np.dot(X, theta)
    errors = predictions - y
    gradient = np.dot(X.T, errors) / m
    theta -= alpha * gradient

# Print the estimated parameters
print("Intercept:", theta[0])
print("Coefficients:", theta[1:])

from sklearn.linear_model import LinearRegression
import numpy as np

# Sample data
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([3, 7, 11])

# Create a Linear Regression model
model = LinearRegression()

# Fit the model using the data
model.fit(X, y)

# Print the estimated parameters
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

#nn

import numpy as np

# Sample data
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([[0], [1], [1]])

# Initialize weights and biases randomly
np.random.seed(42)
W1 = np.random.randn(2, 4)  # Weight matrix of the first layer
b1 = np.random.randn(4)     # Bias vector of the first layer
W2 = np.random.randn(4, 1)  # Weight matrix of the output layer
b2 = np.random.randn(1)     # Bias vector of the output layer

# Hyperparameters
alpha = 0.01  # Learning rate
num_iterations = 1000

# Perform gradient descent
for _ in range(num_iterations):
    # Forward propagation
    Z1 = np.dot(X, W1) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = 1 / (1 + np.exp(-Z2))

    # Backpropagation
    dZ2 = A2 - y
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0)
    dZ1 = np.dot(dZ2, W2.T) * (1 - np.power(A1, 2))
    dW1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1, axis=0)

    # Update weights and biases
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2

# Print the trained weights and biases
print("Weights of the first layer:")
print(W1)
print("Biases of the first layer:")
print(b1)
print("Weights of the output layer:")
print(W2)
print("Biases of the output layer:")
print(b2)

