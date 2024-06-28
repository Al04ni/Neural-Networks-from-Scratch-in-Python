from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
import numpy as np
import nnfs


nnfs.init()

X, y = spiral_data(samples=100, classes=3)

# x axis = X[:, 0]
# y axis = X[:, 1]
# y = classes

plt.scatter(X[:, 0], X[:, 1], c=y, cmap="brg")
plt.show()
