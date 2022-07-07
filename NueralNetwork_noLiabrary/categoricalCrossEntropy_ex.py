'''
1. simple log exapmle 
solving for x in e ** x = b
'''
import numpy as np
import math
b = 5.2

logExample = np.log(b)
print(np.log(b))
print(math.e ** logExample)  
'''
2. Categorical Cross Entropy Example

start with an example output (sofmax_output)

'''
softmax_output = [0.7, 0.1, 0.2]
target_class = 0 
#at position 0 the target is hot
target_output = [1, 0, 0]

#categorical cross entropy loss function 
loss = -(math.log(softmax_output[0])*target_output[0] + 
         math.log(softmax_output[1])*target_output[1] + 
         math.log(softmax_output[2])*target_output[2])
print(loss)
'''
3. Categorical Cross Entropy (with batches)

'''
softmax_ouptuts = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])
class_targets = [0,1,1]
#print the elements at class_targets for each row [0,1,2]
print(softmax_ouptuts[[0,1,2], class_targets])

