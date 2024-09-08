import math

# An example output from the output layer of the neural network
softmax_output = [0.7, 0.1, 0.2]

# Ground truth
target_output = [1, 0, 0]

loss = -(
    math.log(softmax_output[0]) * target_output[0] +
    math.log(softmax_output[1]) * target_output[1] +
    math.log(softmax_output[2]) * target_output[2]
)

print(loss) # 0.35667494393873245

# We can make a few assumptions given one-hot target vectors. 
# First, what are the values for target_output[1] and target_output[2] in this case? 
# They're both 0, and anything multiplied by 0 is 0. Thus, we don't need to calculate these indices. 
# Next, what's the value for target_output [0] in this case? It's 1. 
# So this can be omitted as any number multiplied by 1 remains the same. 
# The same output then, in this example can be calculated with:
loss2 = -math.log(softmax_output[0])

print(loss2) # 0.35667494393873245