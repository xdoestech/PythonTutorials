'''
Guide/ Source: https://www.youtube.com/watch?v=Wo5dMEP_BbI&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3

this file is an overview of how each layer in a nueral network is set up/computed
two layers to show how the entire network is laid out
'''
import numpy as np
#input layer is either the data itself or output from previous layer
inputs = [[1,2,3,2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

#batching
    #grouping data before fitting 
    #better generalizations
    #faster processing

#weights are the strenght of each connection 
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
#layer must correspond to input size
    #layer1 outputs a 3x3 
    #weights 1 must have 3 collums after transpose
weights2 = [[0.1, -0.14, 0.5],
           [-0.5, 0.12, -0.33],
           [-0.44, 0.73, -0.13]]

#bias offset the value some amount 
    #shifts the curve a scalar amount
biases = [2, 3, 0.5]

biases2 = [-1, 2, -0.5]
#output is sum of input*weights + bias for every node 
    #dot product(weights, inputs) + bias
    # need to transpose weights matrix 
    #input rows = weights collums 
    #after dot product add biases 
layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

print(layer2_outputs)