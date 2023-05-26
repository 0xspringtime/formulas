from scipy import stats
import numpy as np

# Sample data
data = np.array([3, 4, 5, 2, 6, 7, 5, 4, 6, 8])

# Parameters
confidence_level = 0.95

# Calculate the confidence interval for the mean
mean, sigma = np.mean(data), np.std(data)
n = len(data)
z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
ci_lower = mean - (z * sigma / np.sqrt(n))
ci_upper = mean + (z * sigma / np.sqrt(n))

# Print the confidence interval
print("Confidence Interval for the Mean (Z-Interval):", ci_lower, ci_upper)

