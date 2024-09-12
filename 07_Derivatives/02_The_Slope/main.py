import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return 2 * x

def f_non_linear(x):
    return 2 * x ** 2

x = np.array(range(5))
# y = f(x)
y = f_non_linear(x)

print(x)
print(y)

# Slope formula - Rate of change between 2 points. 
print((y[1] - y[0]) / (x[1] - x[0])) # 2.0
print((y[2] - y[1]) / (x[2] - x[1])) # 6.0
print((y[3] - y[2]) / (x[3] - x[2])) # 10.0
print((y[4] - y[3]) / (x[4] - x[3])) # 14.0


plt.plot(x, y)
plt.scatter(x, y, color="red", zorder=5)
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)

for i in range(len(x)):
    plt.text(x[i], y[i], f"{x[i], y[i]}", fontsize=9, ha="right", va="bottom")


plt.show()


#       ----- Animations -----
# Parabolic Function
# https://nnfs.io/yup/

# Parabolic function derivatives
# https://nnfs.io/bro/
