import numpy as np
import os
from keras.datasets import mnist

mnistDataSet = mnist.load_data()

# Load the MNIST dataset from mnistDataSet variable
def load_mnist_images():
    (x_train, _), (x_test, _) = mnistDataSet
    x_train = x_train.reshape(-1, 28*28) / 255.0
    x_test = x_test.reshape(-1, 28*28) / 255.0
    return x_train, x_test

def load_mnist_labels():
    (_, y_train), (_, y_test) = mnistDataSet
    return y_train, y_test

def init_params(input_size, hidden_size, output_size):
    W1 = np.random.randn(input_size, hidden_size) * 0.01 
    b1 = np.zeros((1, hidden_size)) 
    W2 = np.random.randn(hidden_size, output_size) * 0.01  
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2

# Forward pass
def relu(x): 
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

