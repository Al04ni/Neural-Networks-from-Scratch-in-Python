import numpy as np


# Softmax activation
class Activation_Softmax:
    # Forward pass
    def forward(self, inputs):
        # Get unnormalized probabilities 
        # Also subtract max value from each input to prevent overflow
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities


# Demonstration of how the exponential function can cause "exploding" values
# aka very large numbers. This would cause havoc in a Neural Network.
print(np.exp(1)) # 2.718281828459045
print(np.exp(10)) # 22026.465794806718
print(np.exp(100)) # 2.6881171418161356e+43
print(np.exp(1000))# RuntimeWarning: overflow encountered in exp
                    # inf

print()
print(np.exp(-np.inf)) # 0.0 exponential of negative infinity approaches zero.
print(np.exp(0)) # 1.0 any number raised to the power of 0 is 1
print(np.exp(np.inf)) # inf

print()
softmax = Activation_Softmax()
softmax.forward([[1, 2, 3]])
print(softmax.output)

print()
# [-2, -1, 0] is a version of the vector above [1, 2, 3] with the max subtracted (3). 
# Both these vectors end up with the exact same output
softmax.forward([[-2, -1, 0]]) 
print(softmax.output)

print()
# This is [1, 2, 3] / 2 
softmax.forward([[0.5, 1, 1.5]])
print(softmax.output)