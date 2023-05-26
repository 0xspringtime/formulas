# Sample data
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Calculate means
mean_x = sum(x) / len(x)
mean_y = sum(y) / len(y)

# Calculate standard deviations
std_dev_x = (sum((xi - mean_x) ** 2 for xi in x) / len(x)) ** 0.5
std_dev_y = (sum((yi - mean_y) ** 2 for yi in y) / len(y)) ** 0.5

# Calculate covariance
covariance = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / len(x)

# Calculate Pearson's r
pearson_r = covariance / (std_dev_x * std_dev_y)

# Print the correlation coefficient
print("Pearson's r:", pearson_r)

from scipy.stats import pearsonr

# Sample data
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Calculate Pearson's r
corr, p_value = pearsonr(x, y)

# Print the correlation coefficient
print("Pearson's r:", corr)
print("pvalue", p_value)


import numpy as np

# Sample data
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Calculate correlation matrix
corr_matrix = np.corrcoef(x, y)

# Extract Pearson's r
corr = corr_matrix[0, 1]

# Print the correlation coefficient
print("Pearson's r:", corr)

