import numpy as np
import math

b = 5.2
print(np.log(b)) # 1.6486586255873816

# Confirming our answer by exponentiating our result above
print(math.e ** 1.6486586255873816) # 5.199999999999999

# log() will always be the Natural Logarithm. 
# Therefore using the base of Eulers numbers
print(math.e) # 2.718281828459045