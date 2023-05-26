import numpy as np

# Example input data
input_data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

# Apply sigmoid activation function
output = 1 / (1 + np.exp(-input_data))

print(output)

import torch
import torch.nn as nn

# Example input data
input_data = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

# Apply sigmoid activation function
sigmoid = nn.Sigmoid()
output = sigmoid(input_data)

print(output)

