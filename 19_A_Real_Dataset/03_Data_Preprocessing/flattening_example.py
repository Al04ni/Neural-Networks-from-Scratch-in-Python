import numpy as np


example = np.array([[1, 2], [3, 4]])
print(example)
"""
[[1 2]
 [3 4]]
"""
print(example.shape) # (2, 2), 2 rows, 2 cols

flattened = example.reshape(-1)
print(flattened) # [1 2 3 4]
print(flattened.shape) # (4,) -> 1D array of length 4