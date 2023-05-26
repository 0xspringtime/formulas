import numpy as np

# Example input data
input_data = np.random.randn(10, 20)

# Apply dropout
p = 0.5  # Dropout probability
mask = np.random.binomial(1, p, size=input_data.shape)
output = (1 / (1 - p)) * mask * input_data

print(output)

import torch
import torch.nn as nn

# Example input data
input_data = torch.randn(10, 20)

# Apply dropout
dropout = nn.Dropout(p=0.5)
output = dropout(input_data)

print(output)

