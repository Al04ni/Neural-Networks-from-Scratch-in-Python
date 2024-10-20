import numpy as np


y_true = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
print(y_true)
print(np.argmax(y_true)) # 0

print(np.argmax(y_true, axis=1)) # [0, 2, 1]