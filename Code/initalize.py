# File: main.py

import numpy as np
import model

if __name__ == "__main__":
    x_train, x_test = model.load_mnist_images()
    y_train_raw, y_test_raw = model.load_mnist_labels()

    input_size = x_train.shape[1]
    hidden_size = 128
    output_size = 10
    learning_rate = 0.1
    epochs = 5
    batch_size = 1

    trained_W1, trained_b1, trained_W2, trained_b2 = model.train_model(
        x_train, y_train_raw,
        x_test, y_test_raw,
        input_size, hidden_size, output_size,
        epochs, learning_rate, batch_size
    )

    print("\nModel training successful!")
    # Example: Make a prediction on the first test image
    # _, prediction_output = model.forward_pass(x_test[0:1], trained_W1, trained_b1, trained_W2, trained_b2)
    # predicted_class = np.argmax(prediction_output)
    # print(f"Prediction for first test image: {predicted_class}, True label: {y_test_raw[0]}")