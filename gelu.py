import math

def gelu(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2))) * x

# Example usage
input_value = 2.0
output_value = gelu(input_value)
print("GELU output:", output_value)

import numpy as np

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

# Example usage
input_array = np.array([1.0, 2.0, 3.0])
output_array = gelu(input_array)
print("GELU output:", output_array)

