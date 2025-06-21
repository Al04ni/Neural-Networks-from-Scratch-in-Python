import matplotlib.pyplot as plt
import numpy as np
import nnfs
import cv2
import os


nnfs.init()
np.set_printoptions(linewidth=200)

# Loads a MNIST dataset
def load_mnist_dataset(dataset, path):
    # Scan all the directories and create a list of labels
    labels = sorted(os.listdir(os.path.join(path, dataset)))

    # Create lists for samples and labels
    X = []
    y = []
    
    # File names
    filenames = []

    # For each label folder
    for label in labels:
        # And for each image in given folder
        for file in sorted(os.listdir(os.path.join(path, dataset, label))):
            # Read the image
            image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)

            # And append it and a label to the lists
            X.append(image)
            y.append(label)
            
            # Store folder + file name
            filenames.append((label, file))

    # Convert the data to proper numpy arrays and return
    return np.array(X), np.array(y).astype('uint8'), filenames


# MNIST dataset (train + test)
def create_data_mnist(path):
    # Load both sets separately
    X, y, filenames = load_mnist_dataset('train', path)
    X_test, y_test, _ = load_mnist_dataset('test', path)
    # And return all the data
    return X, y, X_test, y_test, filenames


directory_path = "../fashion_mnist_images"

# Create dataset
X, y, X_test, y_test, filenames = create_data_mnist(directory_path)

# Scale features
X = (X.astype(np.float32) - 127.5) / 127.5
X_test = (X_test.astype(np.float32) - 127.5) / 127.5


# Reshape to vectors
X = X.reshape(X.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# print(y[0:10]) # [0 0 0 0 0 0 0 0 0 0]
# print(y[6000:6010]) # [1 1 1 1 1 1 1 1 1 1]

keys = np.array(range(X.shape[0]))
# print(keys[:10]) # [0 1 2 3 4 5 6 7 8 9]

np.random.shuffle(keys)
# print(keys[:10]) # [ 3048 19563 58303  8870 40228 31488 21860 56864   845 25770]


X = X[keys]
y = y[keys]
filenames = np.array(filenames)[keys]

print(y[:15])


# T-shirt/top - label 0
# print("Label:", y[8])
# plt.imshow((X[8].reshape(28, 28))) # Reshape as image is a vector already
# plt.show()



# Label map
label_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


# Show the first 15 images with label, ID, and original index
plt.figure(figsize=(12, 6))  # Wider aspect ratio

for i in range(15):
    label = y[i]
    label_text = label_names[label]
    folder, file = filenames[i]

    plt.subplot(3, 5, i + 1)
    # plt.imshow(X[i].reshape(28, 28), cmap='gray')
    plt.imshow(X[i].reshape(28, 28))
    plt.title(f"{label_text} ({label})\n{folder}/{file}", fontsize=7)
    plt.axis('off')


plt.tight_layout()
plt.show()
