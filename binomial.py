import math

# Parameters
n = 10  # Number of trials
p = 0.5  # Probability of success

# Calculate the binomial coefficient
def binomial_coefficient(n, k):
    return math.factorial(n) / (math.factorial(k) * math.factorial(n - k))

# Calculate the PMF for a specific value of k
k = 5  # Specific number of successes
pmf = binomial_coefficient(n, k) * p**k * (1-p)**(n-k)

# Print the PMF
print("PMF (without library):", pmf)

from scipy.stats import binom

# Parameters
n = 10  # Number of trials
p = 0.5  # Probability of success

# Calculate the PMF for a specific value of k
k = 5  # Specific number of successes
pmf = binom.pmf(k, n, p)

# Print the PMF
print("PMF (with library):", pmf)

import numpy as np

# Parameters
n = 10  # Number of trials
p = 0.5  # Probability of success

# Generate the binomial distribution
binomial_dist = np.random.binomial(n, p, size=1000)

# Count occurrences of a specific value of k
k = 5  # Specific number of successes
count = np.sum(binomial_dist == k)

# Calculate the PMF
pmf = count / len(binomial_dist)

# Print the PMF
print("PMF (with library):", pmf)

