# NumberRecognition

A neural network used to recognize numbers, trained on the MNIST dataset. 
This project includes a drawing interface that allows users to draw digits and see the model's predictions in real-time.
Going from a solution using tensorflow frameworks (first commit) to a manually implemented neural network. The project magnifies the normal 28x28 template used which MNIST to better the user expirience. Minimizes it and feeds it into the AI model to make a prediction.  
The accuracy when using tensorflow was about 97% (4 epochs) on the MNIST dataset  
The accuracy when using my implementation 92% (4 epochs, 128 batch size) on the MNIST dataset  


## Features
- AI predicts which number you have drawn on a small canvas
- Press **'p'** to make a prediction.
- Press **'c'** to clear the board.

## Usage

1. Run the drawing interface script. (If you dont have the parameters for the network, run initialize.py first)
2. Draw a digit in the window.
3. Press **'p'** to see the model's prediction on the screen.
4. Press **'c'** to clear and try again. 
5. The model will display the predicted digit

## Errors
You may get a error after running every epoch, it is an iteration error
It is fixable, however the parameters are loaded and usable in the drawinginterface so i didnt mind to do it :D
