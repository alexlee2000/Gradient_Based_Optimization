# Preprocessing
#--------------------------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from numpy import asarray
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
Q2_data = pd.read_csv("Q2.csv")
Q2_data = Q2_data.dropna()
X_Q2_data = Q2_data[["age", "nearestMRT", "nConvenience"]]
Y_Q2_data = Q2_data[["price"]]
scaler = MinMaxScaler()
X_Q2_data = asarray(scaler.fit_transform(X_Q2_data))
Xtrain = asarray(X_Q2_data[:204,:])
Ytrain = asarray(Y_Q2_data.head(204))
Xtest = asarray(X_Q2_data[204:,:])
Ytest = asarray(Y_Q2_data.tail(204))
#--------------------------------------------------------------------------------------------------------------------------------

# Using Jax to implement gradient descent 
import jax.numpy as jnp 
from jax import grad 

def L_w_func(w): # loss for training data
    X = jnp.asarray(Xtrain) # 204 x 3 
    X = np.insert(Xtrain, 0, 1, axis=1) # 204 x 4 (+ 1 col of ones)
    Y = jnp.asarray(Ytrain) # 1 x n 
    Y = jnp.transpose(Y) # n x 1 

    wTx = jnp.transpose(w).dot(jnp.transpose(X))*-1 
    ywTx = jnp.append(wTx, Y, axis=0)
    ywTx = jnp.sum(ywTx, axis = 0)
    ywTx = jnp.square(ywTx)*(1/4)
    ywTx = ywTx + 1 
    ywTx = jnp.sqrt(ywTx)
    ywTx = ywTx - 1 
    return (1/n)*jnp.sum(ywTx) #L(w)
 
def L_w_func_Test(w): # loss for test data 
    X = jnp.asarray(Xtest) # 204 x 3 
    X = np.insert(Xtest, 0, 1, axis=1) # 204 x 4 (+ 1 col of ones)
    Y = jnp.asarray(Ytest) # 1 x n 
    Y = jnp.transpose(Y) # n x 1 

    wTx = jnp.transpose(w).dot(jnp.transpose(X))*-1 
    ywTx = jnp.append(wTx, Y, axis=0)
    ywTx = jnp.sum(ywTx, axis = 0)
    ywTx = jnp.square(ywTx)*(1/4)
    ywTx = ywTx + 1 
    ywTx = jnp.sqrt(ywTx)
    ywTx = ywTx - 1 
    return (1/n)*jnp.sum(ywTx) #L(w)

x_values = np.zeros(1306)
y_values = np.zeros(1306) 

w = jnp.array([[1.0],[1.0],[1.0],[1.0]]) # init w
alpha = 1.0 # init alpha 
converge = False 
k = 0 # iterations
n = 204 # obs 
while (converge == False):
    L_w = L_w_func(w) # L(w)

    L_w_grad = grad(L_w_func) # gradient computation
    w = w - alpha*L_w_grad(w) # w update

    L_w_new = L_w_func(w) # new L(w)

    x_values[k] = k
    y_values[k] = L_w

    k = k + 1
    if(abs(L_w_new - L_w) < 0.0001):
        converge = True
        print("CONVERGED at k =",k)
        print("Final weight vector is:\n",w)
        print("Final Train Loss is :", L_w_new)
        print("Final Test  Loss is :", L_w_func_Test(w))

# plotting loss against iterations k 
plt.plot(x_values, y_values)
plt.title('Training loss at each iteration')
plt.xlabel('k')
plt.ylabel('L(w)')
plt.xticks(np.arange(0, k, 100))
plt.xticks(rotation=45)
plt.show()
