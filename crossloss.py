import numpy as np

# Example data
y_true = np.array([1, 0, 2])  # True class labels
y_pred = np.array([[0.1, 0.8, 0.1], [0.9, 0.2, 0.1], [0.1, 0.3, 0.6]])  # Predicted class probabilities

# Compute Cross-Entropy Loss
N = len(y_true)
C = y_pred.shape[1]
loss = -np.sum(np.log(y_pred[np.arange(N), y_true])) / N
print(loss)

import torch
import torch.nn as nn

# Example data
y_true = torch.tensor([1, 0, 2])  # True class labels
y_pred = torch.tensor([[0.1, 0.8, 0.1], [0.9, 0.2, 0.1], [0.1, 0.3, 0.6]])  # Predicted class probabilities

# Compute Cross-Entropy Loss
loss = nn.CrossEntropyLoss()
output = loss(y_pred, y_true)
print(output.item())

