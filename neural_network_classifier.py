import numpy as np
import math

IMAGES_PATH = 'train-images-idx3-ubyte'
LABELS_PATH = 'train-labels-idx1-ubyte'

# Features are equal to the number of nodes in the input layer
N_FEATURES = 28 * 28
N_CLASSES = 10

x, y = read_mnist(IMAGES_PATH, LABELS_PATH)
x, y = shuffle_data(X, y, random_seed=RANDOM_SEED)
x_train, y_train = X[:500], y[:500]
x_test, y_test = X[500:], y[500:]
