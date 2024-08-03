from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
import numpy as np
import nnfs

nnfs.init()


# Dense layer
class Layer_Dense:
    # Layer Initialization
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # Forward Pass
    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

# ReLU activation
class Activation_ReLU:
    # Forward pass
    def forward(self, inputs):
        # Calculate output values from input
        self.output = np.maximum(0, inputs)

# Softmax activation
class Activation_Softmax:
    # Forward pass
    def forward(self, inputs):
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Create ReLU activation (to be used with Dense layer 1):
activation1 = Activation_ReLU()

# Create second Dense layer with 3 input features (as we take output
# of previous Dense layer here) and 3 output values
dense2 = Layer_Dense(3, 3)

# Create Softmax activation (to be used with Dense layer 2):
activation2 = Activation_Softmax()

# Perform a forward pass of our training data through this layer
dense1.forward(X)

# Forward pass through the activation function.
# It takes the output of the first Dense layer here.
activation1.forward(dense1.output)

# Make a forward pass through second Dense layer.
# It takes the outputs of the activation function of the first Dense layer as inputs.
dense2.forward(activation1.output)

# Make a forward pass through the activation function.
# It takes the output of the second Dense layer here.
activation2.forward(dense2.output)


# Let's see output of the first few samples:
print(activation2.output[:5])