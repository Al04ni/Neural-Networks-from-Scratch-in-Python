import numpy as np


inputs = [1.0, 2.0, 3.0, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2.0

outputs = np.dot(weights, inputs) + bias
outputs_numpy_hardcoded_no_bias = np.dot([0.2, 0.8, -0.5, 1.0], [1.0, 2.0, 3.0, 2.5])
outputs_numpy_hardcoded_with_bias = np.dot([0.2, 0.8, -0.5, 1.0], [1.0, 2.0, 3.0, 2.5]) + bias

outputs_hardcoded_no_bias = 0.2 * 1.0 + 0.8 * 2.0 + -0.5 * 3.0 + 1.0 * 2.5
outputs_hardcoded_with_bias = 0.2 * 1.0 + 0.8 * 2.0 + -0.5 * 3.0 + 1.0 * 2.5 + 2.0

print(outputs) # 4.8
print(outputs_numpy_hardcoded_no_bias) # 2.8
print(outputs_numpy_hardcoded_with_bias) # 4.8

print("-----")

print(outputs_hardcoded_no_bias) # 2.8
print(outputs_hardcoded_with_bias) # 4.8




#       ----- Animations -----
# Using the dot product for a neuron's calculation
# https://nnfs.io/blq/