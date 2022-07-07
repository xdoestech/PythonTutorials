#to get elements from list 
def getEachElement():
    print('-----print each element-----')
    a = [1,2,3]
    for v in a:
        print(v)

def getEachElementAndIndex():
    print('=====each element AND index======')
    a = [1,2,3]
    for i, v in enumerate(a):
        print(f"elment in list: {v}")
        print(f"index of element: {i}")


getEachElement()
getEachElementAndIndex()