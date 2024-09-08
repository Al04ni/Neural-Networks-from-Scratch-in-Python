import numpy as np


print(-np.log(0)) # RuntimeWarning: divide by zero encountered in log inf

print(np.e **(-np.inf)) # 0.0

print(np.mean([1, 2, 3, -np.log(0)])) # RuntimeWarning: divide by zero encountered in log inf

print(-np.log(1e-7)) # 16.11809565095832

print(-np.log(1+1e-7)) # -9.999999505838704e-08

print(-np.log(1-1e-7)) # 1.0000000494736474e-07

y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

#      1e-7        s1 - 1e-7
# [0.0000001000, 0.9999999000]

# np.clip(array, min_value, max_value)
# array: The input array or list.
# min_value: The minimum value to clip to. Any values in the array smaller than this will be set to this value.
# max_value: The maximum value to clip to. Any values in the array larger than this will be set to this value.

