import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FFMpegWriter


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
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases
    
    # Backward Pass
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)


# ReLU activation
class Activation_ReLU:
    
    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from input
        self.output = np.maximum(0, inputs)
    
    # Backward pass
    def backward(self, dvalues):
        # Since we need to modify the original variable,
        # let's make a copy of the values first
        self.dinputs = dvalues.copy()

        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0


# Softmax activation
class Activation_Softmax:
    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs

        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities
    
    # Backward pass
    def backward(self, dvalues):
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


# Common loss class
class Loss:
    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y):

        # Calculate sample losses
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        # Return loss
        return data_loss


# Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):
    # Forward pass
    def forward(self, y_pred, y_true):

        # Number of samples in a batch
        samples = len(y_pred)

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values - only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]
        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )
        
        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)

        return negative_log_likelihoods

    # Backward pass
    def backward(self, dvalues, y_true):

        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(dvalues[0])

        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        
        # Calculate the gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples


# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy():

    # Creates activation and loss function objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()
    
    # Forward pass
    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)
    
    # Backward pass
    def backward(self, dvalues, y_true):
        
        # Number of samples
        samples = len(dvalues)
        # If labels are one-hot encoded, 
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        
        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples


# SGD (Stochastic Gradient Descent) Optimizer
class Optimizer_SGD:
    # Initialize optimizer - set settings,
    # learning rate of 1. is the default for this optimizer
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate

    # Update parameters
    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases


# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 64 output values
dense1 = Layer_Dense(2, 64)

# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()

# Create second Dense layer with 64 input features (as we take output
# of the previous layer here) and 3 output values
dense2 = Layer_Dense(64, 3)

# Create Softmax classifier's combined loss and activation
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# Create optimizer
optimizer = Optimizer_SGD(learning_rate=0.75) # acc: 0.963, loss: 0.099 <--- best 

# Set up the main figure with a GridSpec layout and adjusted width ratios
fig = plt.figure(figsize=(14, 6))  # Increase figure width for more space
gs = GridSpec(2, 2, width_ratios=[2, 1.5], height_ratios=[1, 1], wspace=0.3, hspace=0.4)

# Define subplots: main decision boundary plot on the left, and stacked loss/accuracy on the right
ax_main = fig.add_subplot(gs[:, 0])  # Decision boundary plot spans both rows on the left
ax_loss = fig.add_subplot(gs[0, 1])  # Loss plot on the top right
ax_accuracy = fig.add_subplot(gs[1, 1])  # Accuracy plot on the bottom right

# Initialize titles for the Loss and Accuracy plots
ax_loss.set_title("Loss: N/A")
ax_accuracy.set_title("Accuracy: N/A")

# Configure the main plot (decision boundary)
ax_main.set_title("Decision Boundary")
ax_main.set_xlabel("Feature 1")
ax_main.set_ylabel("Feature 2")

# Configure the loss and accuracy lines
loss_line, = ax_loss.plot([], [], color='brown')  # Initialize an empty line for loss
accuracy_line, = ax_accuracy.plot([], [], color='blue')  # Initialize an empty line for accuracy

# Lists to store loss and accuracy values for plotting
losses = []
accuracies = []


# Training loop
for epoch in range(100_000):
    
    # Forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y)
    
    # Calculate accuracy
    predictions = np.argmax(loss_activation.output, axis=1)
    accuracy = np.mean(predictions == y) * 100  # Convert to percentage

    # Store accuracy and loss for plotting
    losses.append(loss)
    accuracies.append(accuracy)

    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    
    # Update weights and biases
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)

    # Update plots more frequently for smoother video
    if epoch % 20 == 0 or epoch == 100_000 - 1:  # Record every 20 epochs
        
        # Update the decision boundary plot
        ax_main.clear()
        ax_main.scatter(X[:, 0], X[:, 1], c=y, cmap='brg', s=10)

        # Define a grid over the feature space
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))

        # Predict on the grid
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        dense1.forward(grid_points)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        predictions_grid = np.argmax(dense2.output, axis=1)
        predictions_grid = predictions_grid.reshape(xx.shape)

        # Plot decision boundary
        ax_main.contourf(xx, yy, predictions_grid, cmap='brg', alpha=0.3)
        ax_main.set_title(f"Decision Boundary at Epoch {epoch}")
        ax_main.set_xlabel("Feature 1")
        ax_main.set_ylabel("Feature 2")

        # Update the loss and accuracy plot titles with the current values
        ax_loss.set_title(f"Loss: {loss:.3f}")
        ax_accuracy.set_title(f"Accuracy: {accuracy:.2f}%")

        # Update the loss and accuracy plots
        loss_line.set_data(range(len(losses)), losses)
        accuracy_line.set_data(range(len(accuracies)), accuracies)
        
        # Update axis limits to fit the new data
        ax_loss.relim()
        ax_loss.autoscale_view()
        ax_accuracy.relim()
        ax_accuracy.autoscale_view()

        # Redraw the plot
        plt.draw()
        plt.pause(0.001)  # Pause briefly to allow the plot to update

        # Optional: Print the progress
        print(f"Epoch {epoch}: Loss = {loss:.3f}, Accuracy = {accuracy:.2f}%")