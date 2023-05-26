import numpy as np

# Sample true values and predicted values
y_true = np.array([1, 2, 3, 4])
y_pred = np.array([1.2, 1.8, 2.9, 4.2])

mean_y_true = np.mean(y_true)
ss_total = np.sum((y_true - mean_y_true) ** 2)
ss_residual = np.sum((y_true - y_pred) ** 2)
r2 = 1 - (ss_residual / ss_total)
print("R-squared:", r2)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

r2 = r2_score(y_true, y_pred)
print("R-squared:", r2)
