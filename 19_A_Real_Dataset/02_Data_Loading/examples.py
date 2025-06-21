import matplotlib.pyplot as plt
import numpy as np
import os
import cv2


np.set_printoptions(linewidth=200)


labels = sorted(os.listdir("../fashion_mnist_images/train"))
print(labels)

files = sorted(os.listdir("../fashion_mnist_images/train/0"))
print(files[:10])
print(len(files))

# Sneaker
image_data = cv2.imread("../fashion_mnist_images/train/7/0002.png", cv2.IMREAD_UNCHANGED)

print(image_data)
plt.imshow(image_data)
plt.show()

# Coat
image_data = cv2.imread("../fashion_mnist_images/train/4/0011.png", cv2.IMREAD_UNCHANGED)

print(image_data)
# plt.imshow(image_data) # Default
plt.imshow(image_data, cmap="gray") # Grayscale
plt.show()