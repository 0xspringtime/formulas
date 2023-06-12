import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = np.random.randn(hidden_size, input_size)
        self.biases1 = np.zeros((hidden_size, 1))
        self.weights2 = np.random.randn(output_size, hidden_size)
        self.biases2 = np.zeros((output_size, 1))

    def forward(self, X):
        # First hidden layer
        z1 = np.dot(self.weights1, X) + self.biases1
        a1 = self.sigmoid(z1)

        # Output layer
        z2 = np.dot(self.weights2, a1) + self.biases2
        a2 = self.sigmoid(z2)

        return a2

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

# Example usage
mlp = MLP(input_size=2, hidden_size=4, output_size=1)
X = np.array([[0.5], [0.8]])
output = mlp.forward(X)
print("MLP output:", output)


