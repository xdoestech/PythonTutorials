'''
Tensors are a specialized data structure that are very similar to arrays and matrices.
In PyTorch, we use tensors to encode the inputs and outputs of a model, as well as the model’s parameters.
'''
import torch
import numpy as np

#INITIALIZING TENSOR 
print("------------- INITIALIZING TENSOR ------------------")

data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
    #FROM NUMPY ARRAY
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
    #from another tensor 
        #TENSOR OF ALL ONES
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")
        #tensor of random floats
x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")
    #with random or constant values
        #shape is a tuple of tenosr dimensions
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

#TENSOR ATTRIBUTES
print('\n')
print("------------- TENSOR ATTRIBUTES ------------------")
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

#TENSOR OPERATIONS
print('\n')
print("------------- TENSOR OPERATIONS ------------------")
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)
#joining tensors
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)
#matrix multiplication
    # This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)
#dot product
    # This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
#convert single element tensor to number
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

'''
In-place operations store the result into the operand.
 They are denoted by a _ suffix. 
 For example: x.copy_(y), x.t_(), will change x.
'''
print(f"{tensor} \n")
tensor.add_(5)
print(tensor)

'''
Tensors on the CPU and NumPy arrays can share their underlying memory locations
changing one will change the other
'''
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

#A change in the tensor reflects in the NumPy array.
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

#NumPy array to Tensor
n = np.ones(5)
t = torch.from_numpy(n)

#Changes in the NumPy array reflects in the tensor.
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")

