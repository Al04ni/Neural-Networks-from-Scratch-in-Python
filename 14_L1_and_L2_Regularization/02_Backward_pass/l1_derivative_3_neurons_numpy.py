import numpy as np


weights = np.array([[0.2, 0.8, -0.5, 1],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]])

dL1 = np.ones_like(weights)
print(dL1)

dL1[weights < 0] = -1

print(dL1) 

# [[ 1.  1. -1.  1.]
#  [ 1. -1.  1. -1.]
#  [-1. -1.  1.  1.]]