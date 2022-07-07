'''
use transforms to manipulate data 
data needs to be in final processed form required for algorithm

All TorchVision datasets have two parameters
 transform to modify the features
 target_transform to modify the labels - that accept callables containing the transformation logic
'''

# The FashionMNIST features are in PIL Image format
# the labels are integers. For training,
#  we need 
#   the features as normalized tensors,
#   the labels as one-hot encoded tensors. 
# To make these transformations, we use ToTensor and Lambda.


import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    # ToTensor converts a PIL image or NumPy ndarray into a FloatTensor.
    # scales the image’s pixel intensity values in the range [0., 1.]
    transform=ToTensor(),
    # Lambda transforms apply any user-defined lambda function. 
    # Here, we define a function to turn the integer into a one-hot encoded tensor. 
    # It first creates a zero tensor of size 10 (the number of labels in our dataset) and calls scatter_ which assigns a value=1 on the index as given by the label y
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

