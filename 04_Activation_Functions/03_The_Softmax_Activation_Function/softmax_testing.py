
# Values from the previous output when we described what a neural network is
layer_outputs = [4.8, 1.21, 2.385]

# ---------- Important ----------
# Even with an indefinite amount of layer outputs, after exponentiating + normalizing, 
# the sum of all the normalized values will still equal to 1
# import numpy as np
# Randomly generate 50 layer outputs
# layer_outputs = np.random.rand(50)
# ----------------------------------


# e - mathematical constant, we use E here to match a common coding
# style where constants are uppercased
E = 2.71828182846

# Or use math.e
# import math
# print(math.e)

# For each value in a vector, calculate the exponential value
exp_values = []
for output in layer_outputs:
    exp_values.append(E ** output)

print("Exponentiated Values: ")
print(exp_values)

# Now normalize the values
norm_base = sum(exp_values) # We sum all values
norm_values = []
for value in exp_values:
    norm_values.append(value / norm_base)

print("Normalized Exponentiated Values: ")
print(norm_values)

print("Sum of normalized values: ", sum(norm_values))
