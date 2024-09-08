

# class targets:
# 0 = dog
# 1 = cat
# 2 = human

softmax_outputs = [[0.7, 0.1, 0.2],
                   [0.1, 0.5, 0.4],
                   [0.02, 0.9, 0.08]]

# Each entry in class_targets corresponds to the correct answer for each sample in softmax_outputs
# class_targets[0] is for softmax_outputs[0]
# class_targets[1] is for softmax_outputs[1]
# class_targets[2] is for softmax_outputs[2]

# class_targets[i] gives you the correct class index for the i-th sample in softmax_outputs.
# For example: 
# class_targets[0] = 0 means that the correct class for the first sample (softmax_outputs[0]) is class 0 (index 0 in first sample).
# class_targets[1] = 1 means that the correct class for the second sample (softmax_outputs[1]) is class 1 (index 1 in second sample).
# class_targets[2] = 1 means that the correct class for the third sample (softmax_outputs[2]) is class 1 (index 1 in third sample).

class_targets = [0, 1, 1] # dog, cat, cat