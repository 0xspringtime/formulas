# Sample data
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Calculate means
mean_x = sum(x) / len(x)
mean_y = sum(y) / len(y)

# Calculate covariance
covariance = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / (len(x) - 1)

# Print the covariance
print("Covariance (without library):", covariance)
import numpy as np

# Sample data
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Calculate covariance
covariance = np.cov(x, y)[0, 1]

# Print the covariance
print("Covariance (with library):", covariance)

import pandas as pd

# Sample data
data = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [2, 4, 6, 8, 10]})

# Calculate covariance
covariance = data['x'].cov(data['y'])

# Print the covariance
print("Covariance (with library):", covariance)

