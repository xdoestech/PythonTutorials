'''
PolyCode Source: https://www.youtube.com/watch?v=Py4xvZx-A1E
'''
import numpy as np
from sklearn import neural_network

class NeuralNetwork():

    def __init__(self):
        np.random.seed(1)
        #3x1 matrix with values between -1 and 1 with mean of 0
        self.synaptic_weights = 2 * np.random.random((3,1)) - 1
         
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 -x)

    def train(self, training_inputs, training_outputs, training_iterations):

        for iteration in range(training_iterations):

            output = self.think(training_inputs)
            error = training_outputs - output
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
            self.synaptic_weights += adjustments

    def think(self, inputs):

        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))

        return output

if __name__ == "__main__":
    neural_network = NeuralNetwork()

    print("Random synaptic weights: ")
    print(neural_network.synaptic_weights)

    #training data
    training_inputs = np.array([[0,0,1],
                           [1,1,1],
                           [1,0,1],
                           [0,1,1]])

    #training output 
    training_outputs = np.array([[0,1,1,0]]).T#transposed so 4x1 matrix

    training_iterations = 10000
    neural_network.train(training_inputs, training_outputs, training_iterations)

    print("Synaptic weights after training: ")
    print(neural_network.synaptic_weights)

    A = str(input("Input 1: "))
    B = str(input("Input 2: "))
    C = str(input("Input 3: "))

    print("New situation: input data = ", A, B, C)
    print("Ouput data: ")
    print(neural_network.think(np.array([A,B,C])))
