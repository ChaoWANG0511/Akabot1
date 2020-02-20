from numpy import *
a=array([[1 ,2,1],
         [3 ,4,4]])
b=array([[5 ,6,6],
         [6 ,0,8]])
print((a+b))
print(hstack((a, b,a+b)))
print(a.shape)
print(len(a))
print(a.extend(b))