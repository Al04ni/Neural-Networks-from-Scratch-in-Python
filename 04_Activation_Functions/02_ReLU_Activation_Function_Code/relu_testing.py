
inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]

output = []
for i in inputs:
    if i > 0:
        output.append(i)
    else:
        output.append(0)

# List comprehension
output_2 = [i if i > 0 else 0 for i in inputs]

print(output)
print(output_2)

# Using max() function
output_3 = []
for i in inputs:
    output_3.append(max(0, i))

print(output_3)

# List comprehension w/ max()
output_4 = [max(0, i) for i in inputs]

print(output_4)

# Numpy method
import numpy as np

output_np = np.maximum(0, inputs)
print(output_np)