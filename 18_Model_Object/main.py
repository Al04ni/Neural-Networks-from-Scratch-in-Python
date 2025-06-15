from layers import Layer_Dense, Layer_Input, Layer_Dropout

from optimizers import (
    Optimizer_Adam, Optimizer_Adagrad, 
    Optimizer_RMSprop, Optimizer_SGD)

from activation_functions import (
    Activation_ReLU, Activation_Linear, 
    Activation_Sigmoid, Activation_Softmax)

from loss_functions import (
    Loss_MeanSquaredError, Loss_BinaryCrossentropy, 
    Loss_CategoricalCrossentropy, Loss_MeanAbsoluteError,
    Activation_Softmax_Loss_CategoricalCrossentropy)

from nnfs.datasets import sine_data, spiral_data
import matplotlib.pyplot as plt
import numpy as np
import nnfs


nnfs.init()


# Model class
class Model:
    def __init__(self):
        # Create a list of network objects
        self.layers = []
        # Softmax classifier's output object
        self.softmax_classifier_output = None
    
    # Add objects to the model
    def add(self, layer):
        self.layers.append(layer)
    
    # Set loss and optimizer and accuracy
    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy
    
    # Finalize the model
    def finalize(self):
        # Create and set the input layer
        self.input_layer = Layer_Input()

        # Count all the objects
        layer_count = len(self.layers)

        # Initialize a list containing trainable layers:
        self.trainable_layers = []

        # Iterate the objects
        for i in range(layer_count):

            # If it's the first layer,
            # the previous layer object is the input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]
            
            # All layers except for the first and the last
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]
            
            # The last layer - the next object is the loss
            # Also let's save the reference to the last object
            # whose output is the model's output
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]
            
            # If layer contains an attribute called "weights",
            # it's a trainable layer -
            # add it to the list of trainable layers
            # We don't need to check for biases -
            # checking for weights is enough
            if hasattr(self.layers[i], "weights"):
                self.trainable_layers.append(self.layers[i])
        
        # Update loss object with trainable layers
        self.loss.remember_trainable_layers(self.trainable_layers)

        # If output activation is Softmax and loss function is Categorical Cross-Entropy
        # create an object of combined activation and loss function containing faster gradient calculation
        if isinstance(self.layers[-1], Activation_Softmax) and isinstance(self.loss, Loss_CategoricalCrossentropy):
            # Create an object of combined activation and loss functions
            self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossentropy()
    
    # Train the model
    def train(self, X, y, *, epochs=1, print_every=1, validation_data=None):
        # Initialize accuracy object
        self.accuracy.init(y)

        # Main training loop
        for epoch in range(1, epochs + 1):
            
            # Perform the forward pass
            output = self.forward(X, training=True)

            # Calculate loss
            data_loss, regularization_loss = self.loss.calculate(output, y, include_regularization=True)
            loss = data_loss + regularization_loss

            # Get predictions and calculate an accuracy
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y)

            # Perform backward pass
            self.backward(output, y)

            # Optimize (update parameters)
            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()

            # Print a summary
            if not epoch % print_every:
                print(f'epoch: {epoch}, ' +
                      f'acc: {accuracy:.3f}, ' +
                      f'loss: {loss:.3f} (' +
                      f'data_loss: {data_loss:.3f}, ' +
                      f'reg_loss: {regularization_loss:.3f}), ' +
                      f'lr: {self.optimizer.current_learning_rate}')
        
        # If there is the validation data
        if validation_data is not None:

            # For better readability
            X_val, y_val = validation_data
            
            # Perform the forward pass
            output = self.forward(X_val, training=False)
            
            # Calculate the loss
            loss = self.loss.calculate(output, y_val)
            
            # Get predictions and calculate an accuracy
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y_val)
            
            # Print a summary
            print(f'validation, ' +
                  f'acc: {accuracy:.3f}, ' +
                  f'loss: {loss:.3f}')
        
    # Performs forward pass
    def forward(self, X, training):
        # Call forward method on the input layer
        # this will set the output propert that
        # the first layer in "prev" object is expecting
        self.input_layer.forward(X, training)

        # Call forward method of every object in a chain
        # Pass output of the previous object as a parameter
        for layer in self.layers:
            layer.forward(layer.prev.output, training)
        
        # "layer" is now the last object from the list,
        # return its output
        return layer.output

    # Performs backward pass
    def backward(self, output, y):

        # If softmax classifier
        if self.softmax_classifier_output is not None:
            # First call backward method
            # on the combined activation/loss
            # this will set dinputs property
            self.softmax_classifier_output.backward(output, y)

            # Since we'll not call backward method of the last layer
            # which is Softmax activation
            # as we used combined activation/loss
            # object, let's set dinputs in this object
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

            # Call backward method going through
            # all objects but last
            # in reversed order passing dinputs as a parameter
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            
            return

        # First call backward method on the loss
        # this will set dinputs property that the last
        # layer will try to access shortly
        self.loss.backward(output, y)

        # Call backward method going through all the objects
        # in reveresed order passing dinputs as a parameter
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)


# Common accuracy class
class Accuracy:
    # Calculates an accuracy
    # given predictions and ground truth values
    def calculate(self, predictions, y):
        # Get comparison results
        comparisons = self.compare(predictions, y)

        # Calculate an accuracy
        accuracy = np.mean(comparisons)

        # Return accuracy
        return accuracy


# Accuracy calculation for classification model
class Accuracy_Categorical(Accuracy):
    def __init__(self, *, binary=False):
        # Binary mode?
        self.binary = binary
    
    # No initialization is needed
    def init(self, y):
        pass

    # Compares predictions to the ground truth vlaues
    def compare(self, predictions, y):
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y


# Accuracy calculation for regression model
class Accuracy_Regression(Accuracy):
    def __init__(self):
        # Create precision property
        self.precision = None
    
    # Calcualtes precision value
    # based on passed-in ground truth
    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250
    
    # Compares predictions to the ground truth values
    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision


# ------------------------------------------------------------------------   
#                   --------   Regression --------   

# Create dataset
# X, y = sine_data(samples=1000)

# # Instantiate the model
# model = Model()

# # Add layers
# model.add(Layer_Dense(1, 64))
# model.add(Activation_ReLU())
# model.add(Layer_Dense(64, 64))
# model.add(Activation_ReLU())
# model.add(Layer_Dense(64, 1))
# model.add(Activation_Linear())

# # Set loss and optimizer objects
# model.set(
#     loss=Loss_MeanSquaredError(),
#     optimizer=Optimizer_Adam(learning_rate=0.005, decay=1e-3),
#     accuracy=Accuracy_Regression()
# )

# # Finalize the model
# model.finalize()

# # Train the model
# model.train(X, y, epochs=10000, print_every=100)
# ------------------------------------------------------------------------   

# ------------------------------------------------------------------------   
#          --------   Binary Logistic Regression --------   
# Create train and test dataset
# X, y = spiral_data(samples=100, classes=2)
# X_test, y_test = spiral_data(samples=100, classes=2)

# # Reshape labels to be a list of lists
# # Inner list contains one output (either 0 or 1)
# # per each output neuron, 1 in this case
# y = y.reshape(-1, 1)
# y_test = y_test.reshape(-1, 1)

# # Instantiate the model
# model = Model()

# # Add layers
# model.add(Layer_Dense(2, 64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
# model.add(Activation_ReLU())
# model.add(Layer_Dense(64, 1))
# model.add(Activation_Sigmoid())

# # Set loss, optimizer and accuracy objects
# model.set(
#     loss=Loss_BinaryCrossentropy(),
#     optimizer=Optimizer_Adam(decay=5e-7),
#     accuracy=Accuracy_Categorical(binary=True)
# )

# # Finalize the model
# model.finalize()

# # Train the model
# model.train(X, y, validation_data=(X_test, y_test), epochs=10000, print_every=100)
# ------------------------------------------------------------------------ 


# ------------------------------------------------------------------------ 
#                 --------  Multiclass Classification --------   
# Create dataset
X, y = spiral_data(samples=100, classes=3)
X_test, y_test = spiral_data(samples=100, classes=3)

# Instantiate the model
model = Model()

# Add layers
model.add(Layer_Dense(2, 512, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.1))
model.add(Layer_Dense(512, 3))
model.add(Activation_Softmax())

# Set loss, optimizer and accuracy objects
model.set(
    loss=Loss_CategoricalCrossentropy(),
    optimizer=Optimizer_Adam(learning_rate=0.05, decay=5e-5),
    accuracy=Accuracy_Categorical()
)

# Finalize the model
model.finalize()

# Train the model
model.train(X, y, epochs=10000, print_every=100, validation_data=(X_test, y_test))
# ------------------------------------------------------------------------ 