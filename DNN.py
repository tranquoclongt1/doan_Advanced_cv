
# coding: utf-8

# # Build a Deep Neural Network (Binary classification)

# - <b>Step by step: </b>
#     1. Initialize parameters (W: weights, b:bias)
#     2. Forward propagation (Compute linear Z and A: activation value using (reLU and sigmoid)
#     3. Back-propagation (Compute derivative of dZ -> dW, db) using in Gradient Decent
#     4. Update parameters
#     5. Predict

# ### Library:  
# - matplotlib
# - numpy
# - dataframe

# In[5]:


import numpy as np
from scipy.special import expit # deal with large float number on sigmoid function
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')


# ## 1. Initialize Parameters for the whole network

# -<b>INPUT: </b>
#     - layer_dims: dimension of network. [5,2,1] reference #neuron in each layer-> input: 5, hidden: 2, output: 1
# -<b> OUTPUT: </b>
#     - dictionary: 
#         - 'W' + str(l): matrix weight of l-th layer. Ex: 'W2': matrix weight of 2-th layer. 1-st is the first hidden layer.
#         - 'b' + str(l): array of bias.

# In[9]:


def initialize_parameters(layer_dims):
    """
    -INPUT:
        - layer_dims: dimension of network. [5,2,1] reference #neuron in each layer-> input: 5, hidden: 2, output: 1
    -OUTPUT: 
        - dictionary: 
            - 'W' + str(l): matrix weight of l-th layer. Ex: 'W2': matrix weight of 2-th layer. 1-st is the first hidden layer.
            - 'b' + str(l): array of bias.
    """
    parameters = {}
    for l in range(1, len(layer_dims)):
        parameters['W' + str(l)] = np.random.rand(layer_dims[l], layer_dims[l-1])*0.001
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        #print('W' + str(l), ': ', parameters['W'+str(l)].shape)
        #print('b' + str(l), ': ', parameters['b'+str(l)].shape)
        
    return parameters
        


# In[8]:


#parameters = initialize_parameters([2, 7, 1])
#print(parameters)
#print(len(parameters))
#print(parameters['W1'].shape)


# ## 2. Compute Linear Forward

# - Compute Linear Forward: $Z = W*A + b$ 
# - INPUT: 
#     - W: shape - $(n_l, n_{l-1})$ with $n_l$ is the #neurons in l-th layer, $n_{l-1}$ the #neurons in (l-1)-th layer
#     - b: shape - $(n_l, 1)$
# - OUTPUT: 
#     - Z: shape - $(n_l, 1)$

# In[1]:


def linear_forward(A, W, b):
    """
    Return Z and cache
    """
    
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    
    return Z, cache


# ## 3. Linear Activation value

# - Compute Linear Activation value: 
#     - Hidden layer: using ReLU activation function.
#         - $A = \max(0, Z)$
#     - Output layer: using Sigmoid activation function.
#         - $A = \sigma(Z) = \frac{1}{1 + e^{-Z}}$

# In[4]:

def softmax(Z):

    temp1 = np.exp(Z)
    temp2 = np.sum(np.exp(Z), axis=1, keepdims=True)

    return temp1 / temp2


def linear_activation(A_pre, W, b, activation):
    """
    Return activation with the correspondence function
    activation = "relu" or "sigmoid"
    """
    
    # compute the linear forward
    Z, linear_cache = linear_forward(A_pre, W, b)
    
    if(activation == 'relu'):
        A = np.maximum(Z, 0, Z) # relu activation function
    elif(activation == 'sigmoid'):
        A = 1/(1 + np.exp(-Z))  # sigmoid activation function
    elif (activation == 'softmax'):
        A = softmax(Z)
        
    activation_cache = Z
    
    return A, (linear_cache, activation_cache)   


# ## 4. Cost Function

# - Compute the cost base on the sigmoid function (the activation of the output layer).
#     - $ Cost = L(AL, Y) = Y*log(AL) + (1-Y)*(log(1-AL))$

# In[5]:


def compute_cost(AL, Y, parameters):
    """
    Return the loss value between Y predict and Y
    """
    m = AL.shape[1]
    cost = (np.dot(Y, np.log(AL).T) + np.dot(1-Y, np.log(1-AL).T))
    cost = -cost.sum()/m

    #cost = cost.item()
    cost = np.squeeze(cost)
    
    return cost


# ## 5. Linear Backward

# - Compute back-propagation: 
#     -  $dAL = \frac{\sigma{L}}{\sigma{AL}} =- (\frac{y}{AL} - \frac{1-Y}{(1-AL)})$

# In[1]:


def linear_backward(dZ, cache):
    A_pre, W, b = cache
    m = A_pre.shape[1]
    
    dW = np.dot(dZ, A_pre.T)/m
    db = np.sum(dZ, axis=1, keepdims=True)/m
    dA_pre = np.dot(W.T, dZ)
    
    return dA_pre, dW, db


# In[4]:


#def sigmoid_backward(dA, A):
    # derivative of sigmoid: (1-sigmoid(Z))sigmoid(Z)
    #return dA*(1-A)*A

def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    return A


def sigmoid_backward(dA, activation_cache):
    A = sigmoid(activation_cache)
    dZ = dA*(1-A)*A
    return dZ

# In[3]:

def softmax_backward(dA, activation_cache, Y):

    A = softmax(activation_cache)

    SM = A.reshape((-1, 1))
    dZ = np.diag(A) - np.dot(SM, SM.T)

    return dZ


def relu_backward(dA, activation_cache):
    # derivative of relu: f'(x) = 1 if x > 0 else f'(x) = 0
    return dA*((activation_cache>0)*1)


# ## 6. Linear Activation Backward

# In[8]:


def linear_activation_backward(dA, cache, activation, Y=None):
    linear_cache, activation_cache = cache
    if activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
    elif activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
    elif activation == 'softmax':
        dZ = softmax_backward(dA, activation_cache, Y)

    dA_pre, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_pre, dW, db


# ## 7. L-Model forward and backward

# In[10]:


def L_model_forward(X, parameters):
    # compute forward through L layer of network
    caches = []
    A = X
    #print(len(parameters))
    L = len(parameters)//2
    
    for l in range(1, L):
        #print("A"+str(l), ': ', A.shape)
        A_pre = A
        A, cache = linear_activation(A_pre, parameters['W' + str(l)], parameters['b'+str(l)], 'relu')
        caches.append(cache)
    
    AL, cache = linear_activation(A, parameters['W'+str(L)], parameters['b'+str(L)], 'sigmoid')
    caches.append(cache)
    
    return AL, caches


# In[11]:


def L_model_backward(AL, Y, caches):
    # compute backward
    
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    
    # intializing the backpropagation
    dAL = -(np.divide(Y, AL + np.finfo(float).eps) - np.divide(1 - Y, 1 - AL + np.finfo(float).eps))
    
    current_cache = caches[L-1]
    grads['dA'+str(L)], grads['dW'+str(L)], grads['db' + str(L)] = linear_activation_backward(dAL, current_cache, 'sigmoid', Y)

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        grads['dA'+str(l+1)], grads['dW'+str(l+1)], grads['db' + str(l+1)] = linear_activation_backward(grads['dA'+str(l+2)], current_cache, 'relu')
    
    return grads
    


# ## 8. Update parameters

# In[10]:


def update_paramaters(parameters, grads, learning_rate):
    L = len(parameters)//2
    
    for l in range(1, L+1):
        parameters['W'+str(l)] -= (learning_rate*grads['dW'+str(l)])
        parameters['b'+str(l)] -= (learning_rate*grads['db'+str(l)])
    return parameters


# ## 9. L-layer Network

# In[1]:


def L_layer_model(X, Y, layer_dims, learning_rate=0.0075, num_iterations = 3000, print_cost=False):
    
    iteration_cost = []
    
    parameters = initialize_parameters(layer_dims)
    
    for i in range(num_iterations):
        AL, caches = L_model_forward(X.T, parameters)
        
        cost = compute_cost(AL, Y)
        
        grads = L_model_backward(AL, Y, caches)
        
        parameters = update_paramaters(parameters, grads, learning_rate)

        print(type(cost))
        print(cost.shape)
        ## print cost
        print("Cost of iterations %s: %f"%(i+1, cost))
        iteration_cost.append(cost)
        
    return parameters, iteration_cost
        

