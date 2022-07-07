'''
Source: https://www.youtube.com/watch?v=kft1AJ9WVDk
a nueral network with only one middle node 
input > synapses > neuron ---> output
synapses are connections between nodes 
single nueron takes weighted sum of all inputs (input * synapse weight)
weighted sum of all inputs are passed through normalizing function to get output
output will be value between 0 and 1 (sigmoid funciton)
'''
import numpy as np

#define activation function
def sigmoid(x):
    return 1/(1+np.exp(-x))

#for the sigmoid derivative x represents sigmoid function
def sigmoid_derivative(x):
    return x * (1-x)

#training data
training_inputs = np.array([[0,0,1],
                           [1,1,1],
                           [1,0,1],
                           [0,1,1]])

#training output
training_outputs = np.array([[0,1,1,0]]).T#transposed so 4x1 matrix

np.random.seed(1)
#starting weights
    #"strength of connection between nodes"
    #dotproduct(weights, inputs)
    #must be array/tensor/matrix that can be dotted with inputs
synaptic_weights = 2 * np.random.random((3,1)) - 1

print('Random starting synaptic weights: ')
print(synaptic_weights)
'''
training process:
use random weights to get network output
calculate difference between network output and actual ouput
    -Error = network output - actual output
adjust weights to attempt to reduce difference
    -multiply error by input by funciton derivative at output
'''
for iteration in range(2000000):
    input_layer = training_inputs

    outputs = sigmoid(np.dot(input_layer, synaptic_weights))

    error = training_outputs - outputs

    adjustments = error * sigmoid_derivative(outputs)

    synaptic_weights += np.dot(input_layer.T, adjustments)

print('Synaptic weights after training')
print(synaptic_weights)


print('Outputs after training')
print(outputs)
