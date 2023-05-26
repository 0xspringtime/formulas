import numpy as np

# Perform backward propagation using numpy
def backward_propagation_with_numpy(AL, Y, caches):
    grads = {}
    L = len(caches)  # Number of layers
    m = AL.shape[1]  # Number of examples

    # Compute the gradient at the output layer
    dZ = AL - Y

    # Propagate the error gradients backward through the layers
    for l in reversed(range(1, L + 1)):
        A_prev, W, b, Z = caches[l - 1]
        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)
        dZ = dA_prev * (Z > 0)  # ReLU derivative applied element-wise

        # Store the gradients
        grads["dW" + str(l)] = dW
        grads["db" + str(l)] = db

    return grads
print(backward_propagation_with_numpy(grads))
