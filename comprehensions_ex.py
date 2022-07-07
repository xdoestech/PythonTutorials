'''
two ways to loop through list and preform operations
'''
def non_comprehension():
    squares = {}
    for i in range(10):
        squares[i] = i*i
    return squares

def comprehension():
    squares = {i: i*i for i in range(10)}
    return squares

print(non_comprehension())
print(comprehension())

#sort and == to check if lists are identical
nonComp = non_comprehension()
Comp = comprehension()

if(nonComp == Comp):
    print("THEY BOTH WORK!!!")