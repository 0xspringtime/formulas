import numpy as np

# Sample true values and predicted values
y_true = np.array([1, 2, 3, 4])
y_pred = np.array([1.2, 1.8, 2.9, 4.2])

mae = np.mean(np.abs(y_true - y_pred))
print("MAE:", mae)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

mae = mean_absolute_error(y_true, y_pred)
print("MAE:", mae)
