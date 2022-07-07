'''
softmax activation funciton 
    softmax =>  "exponentiate and normalize"
exponentiates output and divides by sum of exponentiated outputs
using ReLU for final output clips a lot of data (all the negatives)
'''
import numpy as np
import nnfs

nnfs.init()

#exapmle output of a layer 
layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.026]]
#numpy will go one item at a time
exp_values = np.exp(layer_outputs)
#we want to sum each row not the entire output
    #sum the layer_ouputs and keep the shape
norm_values = exp_values / np.sum(layer_outputs, axis=1, keepdims=True)
print(np.sum(layer_outputs, axis=1, keepdims=True))
