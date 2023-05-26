import math

# Parameters
mu = 0  # Mean
sigma = 1  # Standard deviation

# Calculate the PDF for a specific value of x
x = 1  # Specific value

# Calculate the PDF
pdf = (1 / (math.sqrt(2 * math.pi * sigma**2))) * math.exp(-(x - mu)**2 / (2 * sigma**2))

# Print the PDF
print("PDF (without library):", pdf)

from scipy.stats import norm

# Parameters
mu = 0  # Mean
sigma = 1  # Standard deviation

# Calculate the PDF for a specific value of x
x = 1  # Specific value
pdf = norm.pdf(x, loc=mu, scale=sigma)

# Print the PDF
print("PDF (with library):", pdf)

import numpy as np

# Parameters
mu = 0  # Mean
sigma = 1  # Standard deviation

# Calculate the PDF for a specific value of x
x = 1  # Specific value
pdf = (1 / (np.sqrt(2 * np.pi * sigma**2))) * np.exp(-(x - mu)**2 / (2 * sigma**2))

# Print the PDF
print("PDF (with library):", pdf)

