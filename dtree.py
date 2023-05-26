#classification
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Sample data
X = np.array([[2, 2], [2, 3], [3, 2], [3, 3]])  # Features
y = np.array([0, 0, 1, 1])  # Class labels

# Create a decision tree classifier
model = DecisionTreeClassifier()

# Fit the model to the data
model.fit(X, y)

# Predict the class labels for new data
X_new = np.array([[2, 2], [2, 3]])
y_pred = model.predict(X_new)

# Print the predicted class labels
print("Predicted class labels:", y_pred)

#regression
from sklearn.tree import DecisionTreeRegressor
import numpy as np

# Sample data
X = np.array([[2], [3], [4], [5]])  # Features
y = np.array([1, 2, 3, 4])  # Target values

# Create a decision tree regressor
model = DecisionTreeRegressor()

# Fit the model to the data
model.fit(X, y)

# Predict the target values for new data
X_new = np.array([[2.5], [3.5]])
y_pred = model.predict(X_new)

# Print the predicted target values
print("Predicted target values:", y_pred)

