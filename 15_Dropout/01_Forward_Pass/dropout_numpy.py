import numpy as np

# If we consider a Bernoulli distribution as a special case of a Binomial distribution with n=1 and look at a list of
# available methods in NumPy, it turns out that there’s a much cleaner way to do this using numpy.
# random.binomial. A binomial distribution differs from Bernoulli distribution in one way, as it adds
# a parameter, n, which is the number of concurrent experiments (instead of just one) and returns the
# number of successes from these n experiments.

# np.random.binomial() works by taking the already discussed parameters n (number of
# experiments) and p (probability of the true value of the experiment) as well as an additional
# parameter size: np.random.binomial(n, p, size).

# The function itself can be thought of like a coin toss, where the result will be 0 or 1. The n is how
# many tosses of the coin do you want to do. The p is the probability for the toss result to be a 1.
# The overall result is a sum of all toss results. The size is how many of these “tests” to run, and the
# return is a list of overall results. For example:

result = np.random.binomial(2, 0.5, size=10)

# This will produce an array that is of size 10, where each element will be the sum of 2 coin tosses,
# where the probability of 1 will be 0.5, or 50%. The resulting array:
print(result) # [1 1 1 0 1 2 0 1 1 1] or other variations


# We can use this to create our dropout layer. Our goal here is to create a filter where the intended
# dropout % is represented as 0, with everything else as 1. For example, let’s say we have a dropout
# layer that we’ll add after a layer that consists of 5 neurons, and we wish to have a 20% dropout.
# An example of a dropout layer might look like:
# [1, 0, 1, 1, 1]

# As you can see, ⅕ of that list is a 0. This is an example of the filter we’re going to apply to
# the output of the dense layer. If we multiplied a neural network’s layer output by this, we’d be
# effectively disabling the neuron at the same index as the 0.

# We can mimic that with np.random.binomial() by doing:
dropout_rate = 0.20
print(np.random.binomial(1, 1 - dropout_rate, size=5))
# [0 1 1 1 1]

# This is based on probabilities, so there will be times when it does not look like the above array.
# There could be times no neurons zero out, or all neurons zero out. On average, these random
# draws will tend toward the probability we desire. Also, this was an example using a very small
# layer (5 neurons). On a realistically sized layer, you should find the probability more consistently
# matches your intended value.

# Assume a neural network layer’s output is:
example_output = np.array([0.27, -1.03, 0.67, 0.99, 0.05,
                           -0.37, -2.01, 1.13, -0.07, 0.73])

# Next, let’s assume our target dropout rate is 0.3, or 30%. We apply a dropout layer:
dropout_rate = 0.3

example_output *= np.random.binomial(1, 1 - dropout_rate, example_output.shape)

print(example_output) # [ 0.27 -1.03  0.    0.    0.05 -0.37 -2.01  1.13 -0.07  0.73]

# Note that our dropout rate is the ratio of neurons we intend to disable (q). Sometimes, the
# implementation of dropout will include a rate parameter that instead means the fraction of
# neurons you intend to keep (p). 
# 
# At the time of writing this, the dropout parameter in deep learning
# frameworks, TensorFlow and Keras, represents the neurons you intend to disable. On the other
# hand, the dropout parameter in PyTorch and the original paper on dropout (http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf) 
# signal the ratio of neurons you intend to keep.

# The way it’s implemented is not important. What is important is that you know which method
# you’re using!

# While dropout helps a neural network generalize and is helpful for training, it’s not something
# we want to utilize when predicting. It’s not as simple as only omitting it because the magnitude
# of inputs to the next neurons can be dramatically different. If you have a dropout of 50%, for
# example, this would suggest that, on average, your inputs to the next layer neurons will be 50%
# smaller when summed, assuming they are fully-connected. What that means is that we used
# dropout during training, and, in this example, a random 50% of neurons output a value of 0 at each
# of the steps. Neurons in the next layer multiply inputs by weights, sum them, and receive values
# of 0 for half of their inputs. If we don’t use dropout during prediction, all neurons will output their
# values, and this state won’t match the state seen during training, since the sums will be statistically
# about twice as big. To handle this, during prediction, we might multiply all of the outputs by
# the dropout fraction, but that’d add another step for the forward pass, and there is a better way
# to achieve this. Instead, we want to scale the data back up after a dropout, during the training
# phase, to mimic the mean of the sum when all of the neurons output their values.

example_output *= np.random.binomial(1, 1 - dropout_rate,example_output.shape) / (1 - dropout_rate)

# Notice that we added the division of the dropout’s result by the dropout rate. Since this rate is a
# fraction, it makes the resulting values larger, accounting for the value lost because a fraction of
# the neuron outputs being zeroed out. This way, we don’t have to worry about the prediction and
# can simply omit the dropout during prediction. In any specific example, you will find that scaling
# doesn’t equal the same sum as before because we’re randomly dropping neurons. That said, after
# enough samples, the scaling will average out overall. To prove this:

print("\n----")
dropout_rate2 = 0.2
example_output2 = np.array([0.27, -1.03, 0.67, 0.99, 0.05,
                          -0.37, -2.01, 1.13, -0.07, 0.73])

print(f'sum initial {sum(example_output2)}')

sums = []
for i in range(10000):

    example_output3 = example_output2 * np.random.binomial(1, 1 - dropout_rate2, example_output2.shape) / (1 - dropout_rate2) 
    sums.append(sum(example_output3))

print(f"mean sum: {np.mean(sums)}")
