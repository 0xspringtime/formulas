import numpy as np

# Sample data
X = np.array([1, 2, 3, 4, 5])  # Independent variable
y = np.array([2, 4, 6, 8, 10])  # Dependent variable

# Calculate the means
mean_x = np.mean(X)
mean_y = np.mean(y)

# Calculate the differences and products
diff_x = X - mean_x
diff_y = y - mean_y
diff_prod = diff_x * diff_y

# Calculate the slope and intercept
slope = np.sum(diff_prod) / np.sum(diff_x**2)
intercept = mean_y - slope * mean_x

# Print the coefficients
print("Intercept:", intercept)
print("Slope:", slope)

from sklearn.linear_model import LinearRegression
import numpy as np

# Sample data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # Independent variable (reshape for a single feature)
y = np.array([2, 4, 6, 8, 10])  # Dependent variable

# Create a linear regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)

# Retrieve the coefficients
intercept = model.intercept_
slope = model.coef_[0]

# Print the coefficients
print("Intercept:", intercept)
print("Slope:", slope)

