
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

# Calculates the gradient of ReLU’s output with respect to z during backpropagation.
# If z > 0, the ReLU derivative is 1, meaning the loss gradient will fully propagate back.
# If z <= 0, the derivative is 0, stopping the gradient from passing back through z, effectively "blocking" updates to earlier weights.
# This calculation uses the chain rule: we multiply the incoming gradient (dvalue) by the local gradient of ReLU to get the overall gradient with respect to z.
drelu_dz = dvalue * (1. if z > 0 else 0.)

print("Gradient of ReLU: ", drelu_dz)

# Applying the chain rule to compute gradients of ReLU output w.r.t each weighted input and bias for backpropagation
dsum_dxw0 = 1
dsum_dxw1 = 1
dsum_dxw2 = 1
dsum_db = 1
drelu_dxw0 = drelu_dz * dsum_dxw0  
drelu_dxw1 = drelu_dz * dsum_dxw1
drelu_dxw2 = drelu_dz * dsum_dxw2
drelu_db = drelu_dz * dsum_db
print("Partial derivatives (gradients) of ReLU output w.r.t xw0, xw1, xw2, and b:", drelu_dxw0, drelu_dxw1, drelu_dxw2, drelu_db)\


# Calculating partial derivatives of ReLU output w.r.t each original input x[i] and weight w[i]
# Here, we apply the chain rule again to backpropagate gradients through the multiplication operations in the forward pass.
dmul_dx0 = w[0]
dmul_dx1 = w[1]
dmul_dx2 = w[2]
dmul_dw0 = x[0]
dmul_dw1 = x[1]
dmul_dw2 = x[2]

drelu_dx0 = drelu_dxw0 * dmul_dx0 
drelu_dw0 = drelu_dxw0 * dmul_dw0

drelu_dx1 = drelu_dxw1 * dmul_dx1
drelu_dw1 = drelu_dxw1 * dmul_dw1

drelu_dx2 = drelu_dxw2 * dmul_dx2
drelu_dw2 = drelu_dxw2 * dmul_dw2

print("Final gradients w.r.t original inputs and weights:", drelu_dx0, drelu_dw0, drelu_dx1, drelu_dw1, drelu_dx2, drelu_dw2)


#       ----- Animations -----
# Backpropagation Example
# https://nnfs.io/pro/