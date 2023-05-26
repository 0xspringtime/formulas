import numpy as np
import matplotlib.pyplot as plt

# Parameters
n = 1000  # Number of samples
sample_size = 100  # Size of each sample
mu = 0  # Mean of the original distribution
sigma = 1  # Standard deviation of the original distribution

# Generate samples from the original distribution
samples = np.random.normal(mu, sigma, size=(n, sample_size))

# Calculate the sample means
sample_means = np.mean(samples, axis=1)

# Plot the histogram of the sample means
plt.hist(sample_means, bins=30, density=True, alpha=0.7, color='blue')

# Calculate the theoretical Gaussian distribution parameters
clt_mu = mu
clt_sigma = sigma / np.sqrt(sample_size)

# Generate x-values for the theoretical Gaussian distribution
x = np.linspace(clt_mu - 4*clt_sigma, clt_mu + 4*clt_sigma, 100)
# Calculate the corresponding y-values using the Gaussian PDF formula
y = 1 / (clt_sigma * np.sqrt(2*np.pi)) * np.exp(-0.5 * ((x - clt_mu) / clt_sigma)**2)
# Plot the theoretical Gaussian distribution
plt.plot(x, y, color='red', lw=2)

plt.xlabel('Sample Means')
plt.ylabel('Density')
plt.title('Central Limit Theorem - Sample Means Distribution')
plt.legend(['Theoretical Gaussian', 'Sample Means'])
plt.show()

