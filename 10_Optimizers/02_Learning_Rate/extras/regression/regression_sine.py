import numpy as np
import nnfs
from nnfs.datasets import sine_data
import matplotlib.pyplot as plt


nnfs.init()

# Create dataset
X, y = sine_data(samples=1000)

# Define a dense layer for regression with a single output
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


# Define ReLU activation (optional for hidden layers)
class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


# Mean Squared Error Loss for regression
class Loss_MeanSquaredError:
    def forward(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        self.dinputs = -2 * (y_true - dvalues) / samples


# SGD (Stochastic Gradient Descent) Optimizer
class Optimizer_SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update_params(self, layer):
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases


# Initialize the layer, activation, loss, and optimizer
dense1 = Layer_Dense(1, 128)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(128, 64)
activation2 = Activation_ReLU()

dense3 = Layer_Dense(64, 1)
loss_function = Loss_MeanSquaredError()
optimizer = Optimizer_SGD(learning_rate=0.02)

# Prepare for plotting
plt.ion()
fig, ax = plt.subplots()

# Scatter initial data points
ax.scatter(X, y, color='blue', label='Data')


# Train model
for epoch in range(23_000):
    # Forward pass through layers
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)

    # Loss calculation
    loss = loss_function.forward(dense3.output, y)

        # if not epoch % 100:
    print(f"epoch: {epoch}, " +
          f"loss: {loss:.3f}, ")

    # Backward pass
    loss_function.backward(dense3.output, y)
    dense3.backward(loss_function.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update weights
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)

    # Plot fitted line every 25 epochs
    if epoch % 25 == 0 or epoch == 23_000 - 1:
        ax.clear()
        ax.scatter(X, y, color='blue', label='Data')
        ax.plot(X, dense3.output, color='red', label='Fitted Line')
        ax.set_title(f"Epoch: {epoch}, Loss: {loss:.4f}")
        plt.legend()
        # plt.pause(0.1)
        plt.pause(0.001)

plt.ioff()
plt.show()
