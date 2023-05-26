import math

# Parameters
lmbda = 2.5  # Average rate or intensity parameter

# Calculate the PMF for a specific value of k
k = 3  # Specific number of events

# Calculate the PMF
pmf = (math.exp(-lmbda) * lmbda**k) / math.factorial(k)

# Print the PMF
print("PMF (without library):", pmf)

import numpy as np

# Parameters
lmbda = 2.5  # Average rate or intensity parameter

# Generate random values from the Poisson distribution
poisson_dist = np.random.poisson(lmbda, size=1000)

# Count occurrences of a specific value of k
k = 3  # Specific number of events
count = np.sum(poisson_dist == k)

# Calculate the PMF
pmf = count / len(poisson_dist)

# Print the PMF
print("PMF (with library):", pmf)

from scipy.stats import poisson

# Parameters
lmbda = 2.5  # Average rate or intensity parameter

# Calculate the PMF for a specific value of k
k = 3  # Specific number of events
pmf = poisson.pmf(k, lmbda)

# Print the PMF
print("PMF (with library):", pmf)

