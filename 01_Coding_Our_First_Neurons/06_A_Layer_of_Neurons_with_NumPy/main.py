import numpy as np

# Output of a layer of 3 neurons. 4 inputs from the previous layer.

inputs = [1.0, 2.0, 3.0, 2.5]
weights = [
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]
biases = [2.0, 3.0, 0.5]

outputs = np.dot(weights, inputs) + biases

outputs_hardcoded = np.array([np.dot(weights[0], inputs), np.dot(weights[1], inputs), np.dot(weights[2], inputs)])
outputs_hardcoded_with_biases = np.array([np.dot(weights[0], inputs), np.dot(weights[1], inputs), np.dot(weights[2], inputs)]) + biases

print(outputs)
print("----")
print(outputs_hardcoded)
print(outputs_hardcoded_with_biases)


#       ----- Animations -----
# Using the dot product with a layer of neurons
# https://nnfs.io/cyx/