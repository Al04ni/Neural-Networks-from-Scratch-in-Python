
weights = [0.2, 0.8, -0.5] # weights of one neuron
dL1 = [] # array of partial derivatives of L1 regularization

for weight in weights:
    if weight >= 0:
        dL1.append(1)
    else:
        dL1.append(-1)

print(dL1) # [1, 1, -1]