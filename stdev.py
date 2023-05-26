# Sample data
data = [1, 2, 3, 4, 5]

# Calculate mean
mean = sum(data) / len(data)

# Calculate squared differences from the mean
squared_diff = [(x - mean) ** 2 for x in data]

# Calculate variance
variance = sum(squared_diff) / len(data)

# Calculate standard deviation
std_dev = variance ** 0.5

# Print the standard deviation
print("Standard Deviation:", std_dev)

import statistics
#uses Bessel's correction

# Sample data
data = [1, 2, 3, 4, 5]

# Calculate standard deviation
std_dev = statistics.stdev(data)

# Print the standard deviation
print("Standard Deviation:", std_dev)

import numpy as np

# Sample data
data = [1, 2, 3, 4, 5]

# Calculate standard deviation
std_dev = np.std(data)

# Print the standard deviation
print("Standard Deviation:", std_dev)

