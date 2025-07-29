import numpy as np
import pickle #Used to save the trained model
from keras.datasets import mnist

mnistDataSet = mnist.load_data()

#helper function to one-hot encode the labels
def one_hot_encode(labels, num_classes):
    num_labels = labels.shape[0]
    one_hot = np.zeros((num_labels, num_classes))
    one_hot[np.arange(num_labels), labels] = 1
    return one_hot
# Load the MNIST dataset from mnistDataSet variable
def load_mnist_images():
    (x_train, _), (x_test, _) = mnistDataSet
    x_train = x_train.reshape(-1, 28*28) / 255.0
    x_test = x_test.reshape(-1, 28*28) / 255.0
    return x_train, x_test

def load_mnist_labels():
    (_, y_train), (_, y_test) = mnistDataSet
    #Remember to one-hot encode the labels
    return y_train, y_test

def init_params(input_size, hidden_size, output_size):
    W1 = np.random.randn(input_size, hidden_size) * 0.01 
    b1 = np.zeros((1, hidden_size)) 
    W2 = np.random.randn(hidden_size, output_size) * 0.01  
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2

#Hidden layer activation function // Use RELU (??)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#Forward pass finding the output from previous layer and input for the next layer
def forward_pass(X,W1,b1,W2,b2):
    z1 = np.dot(X,W1)+ b1 
    a1 = sigmoid(z1) #Compress the matrix multiplication result to a value between 0 and 1
    z2 = np.dot(a1,W2) + b2
    output = softmax(z2)
    return a1,output

#Output layer activation function
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

#Cross-entropy to fine tune the model, calculating the loss
def cross_entropy(y_true, y_pred):
    '''y_pred = softmax(y_pred)
    loss = 0
    for i in range(len(y_pred)):
        loss = loss + (-1*y_true[i]*np.log(y_pred[i]))'''
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
    return loss

# Backward propagation to update the weights and biases
def backward_propagatation(X, y_true, a1, output, W2):
    m = X.shape[0]
    delta2 = output - y_true
    dW2 = np.dot(a1.T, delta2) / m
    db2 = np.sum(delta2, axis=0, keepdims=True) / m

    delta1 = np.dot(delta2, W2.T) * (a1 * (1 - a1)) 
    dW1 = np.dot(X.T, delta1) / m
    db1 = np.sum(delta1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2

def train_model(X_train, y_train, X_test, y_test, input_size, hidden_size, output_size, epochs, learning_rate, batch_size):
    y_train_one_hot = one_hot_encode(y_train, output_size)
    y_test_one_hot = one_hot_encode(y_test, output_size)
    
    #Initialize parameters
    W1,b1,W2,b2 = init_params(input_size, hidden_size, output_size)

    num_training_samples = X_train.shape[0]

    print(f"Starting training with {num_training_samples} samples, {epochs} epochs, batch_size {batch_size}, learning_rate {learning_rate}")

    for epoch in range(epochs):
        permutation = np.random.permutation(num_training_samples)
        x_train_shuffled = X_train[permutation]
        y_train_shuffled_one_hot = y_train_one_hot[permutation]


        for i in range(0, num_training_samples, batch_size):
            x_batch = x_train_shuffled[i:i + batch_size]
            y_batch_true = y_train_shuffled_one_hot[i:i + batch_size]

            #Forward pass batches
            a1,output = forward_pass(x_batch,W1,b1,W2,b2)
            #Calculate loss / backward propagation
            dW1, db1, dW2, db2 = backward_propagatation(x_batch, y_batch_true, a1, output, W2)

            #Update the weights and biases
            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1
            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2

        _, train_output = forward_pass(X_train, W1, b1, W2, b2)
        train_loss = cross_entropy(y_train_one_hot, train_output)
        train_accuracy = np.mean(np.argmax(train_output, axis=1) == y_train)

        _, test_output = forward_pass(X_test, W1, b1, W2, b2)
        test_loss = cross_entropy(y_test_one_hot, test_output)
        test_accuracy = np.mean(np.argmax(test_output, axis=1) == y_test)

        print(f"Epoch {epoch+1}, Batch {i//batch_size+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        
        # Save trained weights to file
        with open("trained_model.pkl", "wb") as f:
            pickle.dump((W1, b1, W2, b2), f)