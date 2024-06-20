import numpy as np


# ---------------------------------------------
# Matrix with a single row. Different ways to implement.
# 1.
a = np.array([[1, 2, 3]])
# print(a)

# 2
b = [1, 2, 3]
# print(np.array([b]))

# 3
c = [1, 2, 3]
# print(np.expand_dims(np.array(c), axis=0))
# ---------------------------------------------


# Matrix product on a row vector and column vector
d = [1, 2, 3]
e = [2, 3, 4]

d = np.array([d])
e = np.array([e]).T # Numpy Matrix Transposition

# Outputs single value in a matrix
print(np.dot(d, e)) 


matrix = [[0.49, 0.97, 0.53, 0.05],
            [0.33, 0.65, 0.62, 0.51],
            [1.00, 0.38, 0.61, 0.45],
            [0.74, 0.27, 0.64, 0.17],
            [0.36, 0.17, 0.96, 0.12]]


# Manual Implementation of Matrix Transposition 
col = 0
transposed_matrix = []

while col < len(matrix[0]):
    new_vector = []
    for vector in matrix:
        new_vector.append(vector[col])
    transposed_matrix.append(new_vector)
    col += 1

print("My algorithm: ")
transposed_matrix_results = np.array(transposed_matrix)
print(transposed_matrix_results)

print("\n ---------------------------- \n")
print("NumPy: ")
numpy_transposed_matrix = np.array(matrix).T
print(numpy_transposed_matrix)


#       ----- Animations -----
# How a transpose works
# https://nnfs.io/qut/

# How a tranpose/transposition works
# https://nnfs.io/pnq/