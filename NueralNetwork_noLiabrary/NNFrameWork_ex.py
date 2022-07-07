'''
network is considered deep learning if there are more than 3 layers
'''
# Source:https://www.youtube.com/watch?v=gmjzbpSVY1A
import re
import numpy as np
import nnfs

from nnfs.datasets import spiral_data
#allows random variables to be reproducible 
# np.random.seed(0)
#below: sets random seed and sets data type for dot product numpy
nnfs.init() #https://github.com/Sentdex/NNfSiX

#inputs
    #Standard practice to use capital 'X' as variable name for inputs 
X = [[1,2,3,2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

#using nnfs liabrary
    #SEE: spiralData.py
X, y = spiral_data(100, 3)
#layer object
    #method to initialize
    #initialize weights and biases
        #weights in range (-1, 1)
        #for this example biases set to zero
    #method to compute output
class Layer_Dense:

    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) #parameters become shape// 0.1 to keep values less than 1
        self.biases = np.zeros((1, n_neurons)) #shape itself is parameter 
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

#ReLU (rectified linear Unit) funciton
    #outputs the maximum of 0 and the input
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
     def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_CategoricalCrossentropy(Loss):
    def forward(self,y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis =1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

X, y = spiral_data(samples=100, classes=3)

#define layers
dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3,3) #the columns and rows of dense1 and dense 2 have to be the same
activation2 = Activation_Softmax()

#pass input through connections + layers
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output) #input for layer is ouput of previous layer
activation2.forward(dense2.output)

print(activation2.output[:5])

loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(activation2.output, y)

print("Loss: ", loss)