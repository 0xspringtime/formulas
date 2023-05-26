import numpy as np

# Sample data
X = np.array([[2.5], [3.0], [4.0], [4.5], [5.0]])  # Independent variable
y = np.array([0, 0, 1, 1, 1])  # Dependent variable (binary)

# Add a constant column to the independent variable
X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

# Initialize parameters
theta = np.zeros(X.shape[1])

# Define the sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define the cost function
def cost_function(theta, X, y):
    m = len(y)
    h = sigmoid(X @ theta)
    cost = (-1/m) * np.sum(y * np.log(h) + (1-y) * np.log(1-h))
    return cost

# Define the gradient function
def gradient(theta, X, y):
    m = len(y)
    h = sigmoid(X @ theta)
    grad = (1/m) * X.T @ (h - y)
    return grad

# Perform gradient descent
alpha = 0.1
iterations = 1000
for _ in range(iterations):
    theta -= alpha * gradient(theta, X, y)

# Retrieve the coefficients
intercept = theta[0]
coefficients = theta[1:]

# Print the coefficients
print("Intercept:", intercept)
print("Coefficients:", coefficients)

from sklearn.linear_model import LogisticRegression
import numpy as np
#applies L2/ridge regularization

# Sample data
X = np.array([[2.5], [3.0], [4.0], [4.5], [5.0]])  # Independent variable
y = np.array([0, 0, 1, 1, 1])  # Dependent variable (binary)

# Create a logistic regression model
model = LogisticRegression()

# Fit the model to the data
model.fit(X, y)

# Retrieve the coefficients
intercept = model.intercept_[0]
coefficients = model.coef_[0]

# Print the coefficients
print("Intercept:", intercept)
print("Coefficients:", coefficients)

