from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Sample data
X = np.array([[1, 'S'], [1, 'M'], [2, 'M'], [3, 'S']])  # Features
y = np.array(['N', 'N', 'Y', 'Y'])  # Class labels

# Encode categorical variables
label_encoders = []
X_encoded = np.empty(X.shape)
for feature_index in range(X.shape[1]):
    le = LabelEncoder()
    X_encoded[:, feature_index] = le.fit_transform(X[:, feature_index])
    label_encoders.append(le)

# Create a Naive Bayes classifier
model = GaussianNB()

# Fit the model to the data
model.fit(X_encoded, y)

# Predict the class labels for new data
X_new = np.array([[2, 'S'], [1, 'M']])
X_new_encoded = np.empty(X_new.shape)
for feature_index in range(X_new.shape[1]):
    le = label_encoders[feature_index]
    X_new_encoded[:, feature_index] = le.transform(X_new[:, feature_index])
y_pred = model.predict(X_new_encoded)

# Print the predicted class labels
print("Predicted class labels:", y_pred)


