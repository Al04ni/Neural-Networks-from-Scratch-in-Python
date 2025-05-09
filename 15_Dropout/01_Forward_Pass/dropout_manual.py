import random


# With the code, we have one hyperparameter for a dropout layer. This is a value for the percentage
# of neurons to disable in that layer. For example, if you chose 0.10 for the dropout parameter, 10%
# of the neurons will be disabled at random during each forward pass.
dropout_rate = 0.5

# Example output containing 10 values
example_output = [0.27, -1.03, 0.67, 0.99, 0.05,
                  -0.37, -2.01, 1.13, -0.07, 0.73]

# Repeat as long as necessary
while True:

    # Randomly choose index and set value to 0
    index = random.randint(0, len(example_output) - 1)
    example_output[index] = 0

    # We might set an index that already is zeroed
    # There are different ways of overcoming this problem,
    # for simplicity we count values that are exactly 0
    # while it's extremely rare in real model that weights
    # are exactly 0, this is not the best method for sure
    dropped_out = 0
    for value in example_output:
        if value == 0:
            dropped_out += 1
    
    # If required number of outputs is zeroed - leave the loop
    if dropped_out / len(example_output) >= dropout_rate:
        break

print(example_output)

# The code is relatively rudimentary, but the idea is to keep zeroing neuron outputs (setting them
# to 0) randomly until we’ve disabled whatever target % of neurons we require.