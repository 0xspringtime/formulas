import numpy as np

# Example data following a power law distribution
alpha = 2.5

# Define the power law distribution using the PDF formula
def power_law_pdf(x, alpha):
    return [alpha * (xi ** (-alpha - 1)) for xi in x]

# Print the PDF values for some example data points
data_points = [1, 2, 3, 4, 5]
pdf_values = power_law_pdf(data_points, alpha)

print("Power Law PDF:")
for x, pdf in zip(data_points, pdf_values):
    print(f"P({x}) = {pdf}")

import scipy.stats as stats

# Example data following a power law distribution
alpha = 2.5

# Define the power law distribution using the Pareto distribution
power_law = stats.pareto(alpha)

# Print the PDF values for some example data points
data_points = [1, 2, 3, 4, 5]
pdf_values = power_law.pdf(data_points)

print("Power Law PDF:")
for x, pdf in zip(data_points, pdf_values):
    print(f"P({x}) = {pdf}")

