import numpy as np

# Example input data
input_data = np.array([1.0, 2.0, 3.0])

# Apply softmax activation function
exponents = np.exp(input_data)
output = exponents / np.sum(exponents)

print(output)

import torch
import torch.nn as nn

# Example input data
input_data = torch.tensor([1.0, 2.0, 3.0])

# Apply softmax activation function
softmax = nn.Softmax(dim=0)
output = softmax(input_data)

print(output)

