import numpy as np
import math

from preprocessing import *
from mathutils import *

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Number of iterations for neural network training
epochs = 50000

input_size, hidden_size, output_size = 2, 3, 1

# Learning rate coeficient
learning_rate = 0.1

# Training data (x = data to train, Y = answers to training data)
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [1]])

# Init the random weights for the input layer to the hidden layer
# and the hidden layer to the output layer
# np.random.uniform returns a 2D array of random numbers from 0.0 to 1.0
w_hidden = np.random.uniform(size=(input_size, hidden_size))
w_output = np.random.uniform(size=(hidden_size, output_size))

# Train the algorithm here
for epoch in range(epochs):

    # Get the dot product of the training data and hidden weights
    hidden_values = np.dot(x, w_hidden)

    # Calculate the sigmoid of the hidden value
    sigmoid_hidden = sigmoid(hidden_values)

    # Get the output by taking the dot product of the sigmoid vector
    # and weight vector for the output
    output = np.dot(sigmoid_hidden, w_output)

    # Calculate the error by substracting the output by the trained data
    error = y - output

    # Backwards propagation
    dZ = error * learning_rate

    # Adjust the weights based on the dot product of the hidden sigmoid vector and the error vector
    # .T stands for transpose
    w_output += sigmoid_hidden.T.dot(dZ)

    # More math the calculate the ajdustment of the hidden layer weights
    dH = dZ.dot(w_output.T) * (sigmoid(sigmoid_hidden) * (1 - sigmoid(sigmoid_hidden)))
    w_hidden += x.T.dot(dH)

    if epoch % 5000 == 0:
        print(sigmoid_hidden.T)
        print(dZ)
        print(sigmoid_hidden.T.dot(dZ))
        print(f'error sum {sum(error)}')


# Testing phase
x_test = x[1]

sh = sigmoid(np.dot(x_test, w_hidden))
print(sh.dot(w_output))
