import cv2 #For computer vision tasks
import numpy as np #For numerical operations and arrays
import matplotlib.pyplot as plt #For plotting and visualizing data
import tensorflow as tf #For building and training neural networks
import os 

#Importing the MNIST dataset and splitting into training and testing data
mnist = tf.keras.datasets.mnist 
trainingData = mnist.load_data()[0]
testData = mnist.load_data()[1] 

(x_train,y_train),(x_test, y_test) = trainingData, testData


x_train = tf.keras.utils.normalize(x_train, axis=1) 
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu')) #Utilizing rectified Linear Unit (RELU)
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax')) #OutPut Layer

#model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
#model.fit(x_train, y_train, epochs=4)
#model.save('number_recognition.keras')

model = tf.keras.models.load_model('number_recognition.keras')
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Loss: {loss}\n Accuracy: {accuracy}")