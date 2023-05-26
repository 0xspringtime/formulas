import numpy as np

# Example input data
input_data = np.random.randn(10, 20)

# Compute batch mean and variance
batch_mean = np.mean(input_data, axis=0)
batch_variance = np.var(input_data, axis=0)

# Apply batch normalization
normalized_data = (input_data - batch_mean) / np.sqrt(batch_variance + 1e-8)
gamma = np.random.randn(20)
beta = np.random.randn(20)
output = gamma * normalized_data + beta

print(output)

import torch
import torch.nn as nn

# Example input data
input_data = torch.randn(10, 20)

# Apply batch normalization
batch_norm = nn.BatchNorm1d(20)
output = batch_norm(input_data)

print(output)

