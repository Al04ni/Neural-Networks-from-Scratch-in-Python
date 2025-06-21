import numpy as np
import cv2
import os


np.set_printoptions(linewidth=200)

# Loads a MNIST dataset
def load_mnist_dataset(dataset, path):
    # Scan all the directories and create a list of labels
    labels = sorted(os.listdir(os.path.join(path, dataset)))

    # Create lists for samples and labels
    X = []
    y = []

    # For each label folder
    for label in labels:
        # And for each image in given folder
        for file in sorted(os.listdir(os.path.join(path, dataset, label))):
            # Read the image
            image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)

            # And append it and a label to the lists
            X.append(image)
            y.append(label)

    # Convert the data to proper numpy arrays and return
    return np.array(X), np.array(y).astype('uint8')


# MNIST dataset (train + test)
def create_data_mnist(path):
    # Load both sets separately
    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)
    # And return all the data
    return X, y, X_test, y_test

directory_path = "../fashion_mnist_images"
# Create dataset
X, y, X_test, y_test = create_data_mnist(directory_path)

# Scale features
X = (X.astype(np.float32) - 127.5) / 127.5
X_test = (X_test.astype(np.float32) - 127.5) / 127.5

print(X.min(), X.max()) # -1.0 1.0

print(X.shape) # (60000, 28, 28) -> 60,000 samples of 28x28 (rows x cols) images

# Reshape to vectors
X = X.reshape(X.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

print(X.shape) # (60000, 784), 60,000 samples of 1D arrays of length 784
print(X[0])