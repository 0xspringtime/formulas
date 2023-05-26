import numpy as np

# Example data for two independent samples
sample1 = np.array([1, 2, 3, 4, 5])
sample2 = np.array([2, 4, 6, 8, 10])

# Calculate means and standard deviations
mean1 = np.mean(sample1)
mean2 = np.mean(sample2)
std1 = np.std(sample1)
std2 = np.std(sample2)

# Calculate sizes of the samples
n1 = len(sample1)
n2 = len(sample2)

# Calculate t-statistic
t_statistic = (mean1 - mean2) / np.sqrt((std1 ** 2 / n1) + (std2 ** 2 / n2))

print("T-Statistic:", t_statistic)

import scipy.stats as stats

# Example data for two independent samples
sample1 = [1, 2, 3, 4, 5]
sample2 = [2, 4, 6, 8, 10]

# Perform independent samples t-test
t_statistic, p_value = stats.ttest_ind(sample1, sample2)

print("T-Statistic:", t_statistic)
print("P-Value:", p_value)

