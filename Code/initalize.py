# File: main.py

import numpy as np
import model
import trainModel
if __name__ == "__main__":
    x_train, x_test = model.load_mnist_images()
    y_train_raw, y_test_raw = model.load_mnist_labels()

    #Play around with the parameters here
    input_size = x_train.shape[1]
    hidden_size = 128
    output_size = 10
    learning_rate = 0.05
    epochs = 20 #Consider implementing early stopping
    batch_size = 32

    trained_W1, trained_b1, trained_W2, trained_b2 = trainModel.train_model(
        x_train, y_train_raw,
        x_test, y_test_raw,
        input_size, hidden_size, output_size,
        epochs, learning_rate, batch_size
    )

    print("\nModel training successful!")
