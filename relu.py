# Sample input
x = [-2, -1, 0, 1, 2]

# Apply ReLU without libraries
relu_output = [max(0, val) for val in x]

# Print the ReLU output
print(relu_output)

import numpy as np

# Sample input
x = np.array([-2, -1, 0, 1, 2])

# Apply ReLU using numpy
relu_output = np.maximum(0, x)

# Print the ReLU output
print(relu_output)

