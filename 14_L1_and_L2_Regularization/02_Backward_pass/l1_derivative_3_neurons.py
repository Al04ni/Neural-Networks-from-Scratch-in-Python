
weights = [[0.2, 0.8, -0.5, 1], # now we have 3 sets of weights
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]
]
dL1 = [] # array of partial derivatives of L1 regularization

for neuron in weights:
    neuron_dL1 = [] # derivatives related to one neuron
    for weight in neuron:
        if weight >= 0:
            neuron_dL1.append(1)
        else:
            neuron_dL1.append(-1)
    dL1.append(neuron_dL1)

print(dL1) # [[1, 1, -1, 1], [1, -1, 1, -1], [-1, -1, 1, 1]]