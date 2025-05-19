import numpy as np


# Dropout
class Layer_Dropout:
    # Init
    def __init__(self, rate):
        # Store the dropout rate, we invert it, for example for dropout
        # of 0.1 we need success rate of 0.9
        self.rate = 1 - rate
    
    # Forward pass
    def forward(self, inputs):
        # Save input values
        self.inputs = inputs
        # Generate and save scaled mask
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        # Apply mask to output values
        self.output = inputs * self.binary_mask

    # Backward pass
    def backward(self, dvalues):
        # Gradients on values
        self.dinputs = dvalues * self.binary_mask