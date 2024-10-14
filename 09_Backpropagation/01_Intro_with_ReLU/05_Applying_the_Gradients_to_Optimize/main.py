
# Forward Pass
x = [1.0, -2.0, 3.0] # input values
w = [-3.0, -1.0, 2.0] # weights
b = 1.0 # bias

# Multiplying inputs by weights
xw0 = x[0] * w[0] # 1.0 * -3.0
xw1 = x[1] * w[1] # -2.0 * -1.0
xw2 = x[2] * w[2] # 3.0 * 2.0

print("Inputs * weights: ", xw0, xw1, xw2) # -3.0, 2.0, 6.0

# Adding weighted inputs and a bias
z = xw0 + xw1 + xw2 + b
print("sum(Inputs * weights + bias): ", z) # 6.0

# ReLU activation function
y = max(z, 0)
print("ReLU output: ", y) # 6.0

# Backward Pass

# The derivative from the next layer
dvalue = 1.0

# Simplified gradient calculation using the chain rule
# Gradient of ReLU output with respect to each input and weight
drelu_dx0 = dvalue * (1. if z > 0 else 0.) * w[0]
drelu_dx1 = dvalue * (1. if z > 0 else 0.) * w[1]
drelu_dx2 = dvalue * (1. if z > 0 else 0.) * w[2]
drelu_dw0 = dvalue * (1. if z > 0 else 0.) * x[0]
drelu_dw1 = dvalue * (1. if z > 0 else 0.) * x[1]
drelu_dw2 = dvalue * (1. if z > 0 else 0.) * x[2]
drelu_db = dvalue * (1. if z > 0 else 0.)

print("Final gradients w.r.t original inputs and weights:", drelu_dx0, drelu_dw0, drelu_dx1, drelu_dw1, drelu_dx2, drelu_dw2)
print("Gradient w.r.t. bias: ", drelu_db)

dx = [drelu_dx0, drelu_dx1, drelu_dx2] # Gradients on inputs
dw = [drelu_dw0, drelu_dw1, drelu_dw2] # Gradients on weights
db = drelu_db # Gradient on bias, just 1 bias here

print("\nCurrent Weights: ", w, "\nCurrent Bias: ", b)

# Applying a fraction of the gradients to current weights and bias
w[0] += -0.001 * dw[0]
w[1] += -0.001 * dw[1]
w[2] += -0.001 * dw[2]
b += - 0.001 * db

print("\nModified Weights: ", w, "\nModified Bias: ", b)

# Another Forward pass

# Multiplying inputs by weights
xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]
print("\nInputs * new weights: ", xw0, xw1, xw2)

# Adding weighted inputs and a bias
z = xw0 + xw1 + xw2 + b
print("sum(Inputs * new weights + new bias): ", z)

# # ReLU activation function
y = max(z, 0)
print("ReLU output: ", y) 


#       ----- Animations -----
# Simplifying Neuron Derivative
# https://nnfs.io/com/