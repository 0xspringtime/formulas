import numpy as np

# Sample true values and predicted values
y_true = np.array([1, 2, 3, 4])
y_pred = np.array([1.2, 1.8, 2.9, 4.2])

mse = np.mean((y_true - y_pred) ** 2)
print("MSE:", mse)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Sample true values and predicted values
y_true = np.array([1, 2, 3, 4])
y_pred = np.array([1.2, 1.8, 2.9, 4.2])

mse = mean_squared_error(y_true, y_pred)
print("MSE:", mse)
