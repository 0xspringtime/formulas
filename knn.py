#classification
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Sample data
X = np.array([[2, 2], [2, 3], [3, 2], [3, 3]])  # Features
y = np.array([0, 0, 1, 1])  # Class labels

# Create a K-NN classifier
model = KNeighborsClassifier(n_neighbors=3)

# Fit the model to the data
model.fit(X, y)

# Predict the class label for new data
X_new = np.array([[2, 2], [2, 3]])
y_pred = model.predict(X_new)

# Print the predicted class labels
print("Predicted class labels:", y_pred)

#regression
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

# Sample data
X = np.array([[2], [3], [4], [5]])  # Features
y = np.array([1, 2, 3, 4])  # Target values

# Create a K-NN regressor
model = KNeighborsRegressor(n_neighbors=3)

# Fit the model to the data
model.fit(X, y)

# Predict the target values for new data
X_new = np.array([[2.5], [3.5]])
y_pred = model.predict(X_new)

# Print the predicted target values
print("Predicted target values:", y_pred)

