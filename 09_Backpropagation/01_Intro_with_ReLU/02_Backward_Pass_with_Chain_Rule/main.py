
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