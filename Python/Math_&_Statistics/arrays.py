import numpy as np

array_1=np.array([1,2,3,4,5])   

print(array_1)

#Array Operations

a=np.array([1,2,3,4,5])
b=np.array([6,7,8,9,10])

print(a+b)
np.savetxt('data.txt', a, delimiter=',')