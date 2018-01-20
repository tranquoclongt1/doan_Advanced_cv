
# coding: utf-8

# In[1]:


import numpy as np
import math


# In[2]:


from Load_Data import LoadData

## Model: vgg16 for vgg16 on Caltech101 -- with vgg16 using data 'db1' 'db2' 'db3'
## Model: caltech5-vgg16 for caltech4 --- with caltech5-vgg16 using data 'db5' and 'db6'
X_train, X_test, Y_a_train, Y_a_test = LoadData().loadData(data='db1', model='vgg16')


# In[3]:


classes = np.unique(Y_a_train)
Y_train = np.array([classes.tolist().index(x) for x in Y_a_train])

classes2 = np.unique(Y_a_test)
Y_test = np.array([classes2.tolist().index(x) for x in Y_a_test])


# In[4]:


classes = np.unique(Y_a_train)
print(classes)


# In[63]:


train_data = np.column_stack((X_train, Y_train))


# In[64]:


# Creat dictinary: key: class --> feature vectors X in class
def creat_dict_data(data):
    
    dict_data = {}
    
    for sample in data:
        
        if str(sample[-1]) not in dict_data:
            dict_data[str(sample[-1])]=[]
            
        dict_data[str(sample[-1])].append(sample[0:-1])
        
    return dict_data


# In[65]:


# Tinh P(c) voi c thuoc 1,2,..,C
def probability_class(dict_data, length):
    
    prob_class = {}
    
    for c in dict_data:
        prob_class[c]=np.log(len(dict_data[c])/length)
        
    return prob_class


# In[66]:


# Compute mean and standard deviation of each class following attribute
def compute_mean_stdev(dict_data):
    
    mean_stdev = {}
    
    for c, values in dict_data.items():
        
        mean = np.mean(values, axis = 0)
        stdev = np.std(values, axis = 0) + np.finfo(float).eps
        
        tup = np.array(tuple(zip(mean, stdev)))
        
        mean_stdev[c] = tup
        
    return mean_stdev


# In[67]:


# Compute p(c|x)
def probability_c_given_x(new_point, mean_stdev, prob_c):
    
    # p(x|c)
    prob_x_given_c = {}
    
    for c in mean_stdev:
    
        prob_1 = np.exp(-(np.power(new_point - mean_stdev[c][:,0],2))/(2*np.power(mean_stdev[c][:,1],2)))
        prob_2 = np.sqrt(2*math.pi*np.power(mean_stdev[c][:,1],2))
        
        p_a = - 0.5*np.log(2*np.pi*np.power(mean_stdev[c][:,1],2))
        p_b = - 0.5*np.log(np.exp(1))*np.power(new_point - mean_stdev[c][:,0],2)/np.power(mean_stdev[c][:,1],2)
        
        prob = np.sum(p_a + p_b)
        
        prob_x_given_c[c] = prob
    
    
    # p(c|x)
    prob_c_given_x = {}
    
    for c in prob_c:
        prob_c_given_x[c] = prob_c[c] + prob_x_given_c[c]
    
    return prob_c_given_x


# In[68]:


# Predict a new_point depend on p(c|x)
def predict_new_point( prob_c_given_x ):
    
    label = max(prob_c_given_x, key = lambda i: prob_c_given_x[i])
    
    return int(float(label))


# In[69]:


# Predict test_data
def predict_GNB(test_data, data):
    
    dict_data = creat_dict_data(data)
    mean_stdev = compute_mean_stdev(dict_data)
   # print(mean_stdev)
    prob_c = probability_class(dict_data, data.shape[0])
    
    labels = []
    
    count = 0
    for sample in test_data:
        
        prob_c_given_x = probability_c_given_x(sample, mean_stdev, prob_c)
        label = predict_new_point(prob_c_given_x)
        labels.append(label)
        count += 1
        if(count%100 == 0):
            print(count, end='\r')
        
    return labels


# In[1]:


labels = predict_GNB(X_test, train_data)


# In[72]:
print("True: ", ((labels==Y_test)*1).sum())
print("Second: ")
print(np.count_nonzero(np.equal(np.array(labels), Y_test)))

