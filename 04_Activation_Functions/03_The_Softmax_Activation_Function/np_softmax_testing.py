import numpy as np


# Values from the previous output when we described what a neural network is
layer_outputs = [4.8, 1.21, 2.385]

# For each value in a vector, calculate the exponential value
exp_values = np.exp(layer_outputs)
print("Exponential Values: ")
print(exp_values)

# Now normalize the values
norm_values = exp_values / np.sum(exp_values)
print("\nNormalized Exponentiated Values: ")
print(norm_values)
print("Sum of Normalized Values: ", np.sum(norm_values))

# To train in batches, we need to convert this functionality to accept layer outputs in batches.
exp_values_2 = np.exp(layer_outputs)
probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)