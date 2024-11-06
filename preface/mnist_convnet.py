"""
    Creating a convolutional neural network to classify MNIST images.
"""

import os

os.environ["KERAS_BACKEND"] = "torch"

import keras
import numpy as np

print(keras.__version__)

from matplotlib import pyplot as plt

with np.load("mnist.npz", allow_pickle=True) as f:
    train_X, train_y = f["x_train"], f["y_train"]
    test_X, test_y = f["x_test"], f["y_test"]

print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  ' + str(test_X.shape))
print('Y_test:  ' + str(test_y.shape))

plt.figure(figsize=(15, 5))
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(train_X[i], cmap=plt.get_cmap('gray'))
plt.show()
