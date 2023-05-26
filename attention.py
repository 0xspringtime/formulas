import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# Example input data
query = np.random.randn(10)  # Query vector
key_value = np.random.randn(5, 10)  # Key-Value matrix

# Compute attention scores
scores = np.dot(key_value, query)
attention_weights = softmax(scores)

# Compute weighted sum
weighted_sum = np.dot(attention_weights, key_value)

print(weighted_sum)

import torch
import torch.nn as nn

# Example input data
query = torch.randn(5, 1, 10)  # Query tensor
key_value = torch.randn(5, 5, 10)  # Key-Value tensor

# Apply attention mechanism
attention = nn.MultiheadAttention(embed_dim=10, num_heads=1)
output, _ = attention(query, key_value, key_value)

print(output)

