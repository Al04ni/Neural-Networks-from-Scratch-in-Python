import numpy as np


softmax_output = [0.7, 0.1, 0.2]

softmax_output = np.array(softmax_output).reshape(-1, 1)

print(np.eye(softmax_output.shape[0])) # shape[0] = 3

print(softmax_output * np.eye(softmax_output.shape[0]))

# diagflat method creates an array using an input vector as the diagonal, a bit faster
print(np.diagflat(softmax_output))

print(np.dot(softmax_output, softmax_output.T))

print(np.diagflat(softmax_output) - np.dot(softmax_output, softmax_output.T))