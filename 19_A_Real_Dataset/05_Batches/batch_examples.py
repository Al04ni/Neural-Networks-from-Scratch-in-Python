import nnfs
from nnfs.datasets import spiral_data

# batch_size = 2
# X = [1, 2, 3, 4]

# print(len(X) // batch_size)

# Create dataset
X, y = spiral_data(samples=100, classes=3)

EPOCHS = 10
BATCH_SIZE = 128 # We take 128 samples at once

# Calculate number of steps
steps = X.shape[0] // BATCH_SIZE

# Dividing rounds down. If there are some remaining data,
# but not a full batch, this won't include it.
# Add 1 to include the remaining samples in 1 more step.
if steps * BATCH_SIZE < X.shape[0]:
    steps += 1