
# coding: utf-8

# # Basic Deep Neural Network on Caltech 101

# In[1]:


import glob
import numpy as np
from scipy import misc





# ### Extract raw pixel as features ### NOTE: RUN ONLY ONCE
# 
# - Extract features and store it to /features/raw_pixel/image_name.pkl
# - <i> To run this line of codes: <b> need to install Keras </b> </i>
# 
# - Using image size 224x224x3 -> give a feature vector length 150528 ** High-dimensional feature vector

# In[2]:

from Load_Data import LoadData

X_train, X_test, Y_train, Y_test = LoadData().loadData(data='db5', model='caltech5-vgg16')


print("#Train: ", len(X_train))
print("#Test: ", len(X_test))

print("Feature Shape: ", X_train[0].shape)

classes = np.unique(Y_test)
print("%s classes: "%(len(classes)))
print(classes)


# just classify two objects
Y_temp_train = np.array([1 if x in classes[0:len(classes)//2] else 0 for x in Y_train])
Y_temp_test = np.array([1 if x in classes[0:len(classes)//2] else 0 for x in Y_test])

from sklearn.preprocessing import LabelBinarizer

mlb = LabelBinarizer()
mlb.fit(Y_train)
Y_train_binarizer = mlb.transform(Y_train)
Y_test_binarizer = mlb.transform(Y_test)



print("Shape of X_train: %s\nShape of Y_train: %s"%(X_train.shape, Y_temp_train.shape))
print("Shape of X_test: %s\nShape of Y_test: %s"%(X_test.shape, Y_temp_test.shape))



len(mlb.classes_)


import DNN
layer_dims = [X_train.shape[1], 2048, len(classes)]


parameters, cost = DNN.L_layer_model(X_train, Y_train_binarizer.T, layer_dims, learning_rate=0.1, num_iterations=1000, print_cost=True)


