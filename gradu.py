# Define the initial parameters
theta = 0.0

# Define the learning rate
learning_rate = 0.1

# Define the gradient of the objective function
gradient = 1.0

# Perform the gradient update
theta_updated = theta - learning_rate * gradient

# Print the updated parameters
print("Updated theta:", theta_updated)


import numpy as np

# Define the initial parameters as NumPy arrays
theta = np.array([0.0, 0.0])

# Define the learning rate
learning_rate = 0.1

# Define the gradient of the objective function as a NumPy array
gradient = np.array([1.0, -0.5])

# Perform the gradient update
theta_updated = theta - learning_rate * gradient

# Print the updated parameters
print("Updated theta:", theta_updated)

import torch

# Define the initial parameters as torch tensors
theta = torch.tensor(0.0, requires_grad=True)

# Define the learning rate
learning_rate = 0.1

# Define the optimizer
optimizer = torch.optim.SGD([theta], lr=learning_rate)

# Perform the gradient update
optimizer.zero_grad()  # Clear the gradients
loss = (theta - 2) ** 2  # Define the loss function
loss.backward()  # Compute the gradients
optimizer.step()  # Update the parameters

# Print the updated parameter
print("Updated theta:", theta.item())
