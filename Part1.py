import numpy as np
from numpy import array
from numpy.linalg import norm

A = np.mat([[1,0,1,-1],
            [-1,1,0,2], 
            [0,-1,-2,1]])

A_t = np.transpose(A)
        
b = np.mat([[1], 
            [2],
            [3]])

x_k_prev = np.mat([[1],
                   [1],
                   [1],
                   [1]])
alpha = 0.1
k = 0
converge = False
while (converge == False):
    if(norm(A_t.dot(A.dot(x_k_prev) - b)) < 0.001):
        converge = True
    if(0 <= k <= 4 or 218 <= k <= 222): #print at first 5 iteration and last 5 iterations
        print("k =", k,", x_k = [", x_k_prev[0,0],",",x_k_prev[1,0],",",x_k_prev[2,0],",",x_k_prev[3,0],"]")
    x_k = x_k_prev - alpha*(A_t.dot(A.dot(x_k_prev) - b))
    #x_k = np.round(x_k,8)
    x_k_prev = x_k   
    k = k + 1

    
