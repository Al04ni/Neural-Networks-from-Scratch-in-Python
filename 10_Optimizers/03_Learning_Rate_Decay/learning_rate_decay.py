
# 1 Step
starting_learning_rate = 1.
learning_rate_decay = 0.1
step = 1

learning_rate_1_step = starting_learning_rate * (1. / (1 + learning_rate_decay * step))

print(learning_rate_1_step) # 0.9090909090909091


# 20 Steps
starting_learning_rate = 1.
learning_rate_decay = 0.1
step = 20

learning_rate_20_steps = starting_learning_rate * (1. / (1 + learning_rate_decay * step))

print(learning_rate_20_steps, "\n") # 0.3333333333333333


# Loop Method
starting_learning_rate = 1.
learning_rate_decay = 0.1

for step in range(21):
    learning_rate_loop = starting_learning_rate * (1. / (1 + learning_rate_decay * step))

    print(learning_rate_loop)

