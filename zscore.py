import numpy as np

# Example input data
data = np.array([1, 2, 3, 4, 5])

# Calculate Z-scores
mean = np.mean(data)
std = np.std(data)
z_scores = (data - mean) / std

print(z_scores)

import scipy.stats as stats

# Example input data
data = [1, 2, 3, 4, 5]

# Calculate Z-scores
z_scores = stats.zscore(data)

print(z_scores)

