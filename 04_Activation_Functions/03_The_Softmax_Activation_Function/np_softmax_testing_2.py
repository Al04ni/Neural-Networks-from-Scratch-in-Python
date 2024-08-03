import numpy as np


layer_outputs = np.array([
                [4.8, 1.21, 2.385],
                [8.9, -1.81, 0.2],
                [1.41, 1.051, 0.026]])


print("Sum without axis")
print(np.sum(layer_outputs))

print("\nThis will be identical to the above since default is None:")
print(np.sum(layer_outputs, axis=None))

# In 2D array (matrix)
# axis 0 = rows
# axis 1 = columns

# Not what we want but just testing
print("\nSumming across rows (axis=0) results in column-wise sums:")
print(np.sum(layer_outputs, axis=0)) 
# 4.8 + 8.9 + 1.41 = 15.11
# 1.21 + -1.81 + 0.2 = 0.451
# 2.385 + 0.2 + 0.026 = 2.611


print("\nBut we want to sum the rows instead, like this w/ raw py: ")
for i in layer_outputs:
    print(sum(i))
# 4.8 + 1.21 + 2.385 = 8.395
# 8.9 + -1.81 + 0.2 = 7.29
# 1.41 + 1.051 + 0.026 = 2.487


print("\nSumming across columns (axis=1) results in row-wise sums:")
print("So we can sum axis 1, but note the current shape: ")
print(np.sum(layer_outputs, axis=1))
# [8.395, 7.29, 2.487]

print("\nSum axis 1, but keep the same dimensions as input:")
print(np.sum(layer_outputs, axis=1, keepdims=True))

shape_check = np.sum(layer_outputs, axis=1, keepdims=True)

print(shape_check.shape)
# [[8.395]
#  [7.29 ]
#  [2.487]]
