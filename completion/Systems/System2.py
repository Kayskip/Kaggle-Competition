#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from scipy.io import arff
get_ipython().run_line_magic('matplotlib', 'inline')


# In[48]:


dataset = pd.read_csv('/home/karu/Documents/COMP309/Kaggle-Competition/completion/Pre-Processed Datasets/Challange-WB-T1_scikit.csv')


# In[49]:


dataset.shape


# In[50]:


dataset.describe()
dataset.head()


# In[63]:


X = dataset['X'].values.reshape(-1,1)
y = dataset['Time'].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()  
regressor.fit(X_train, y_train) #training the algorithm


# In[ ]:





# In[64]:


#To retrieve the intercept:
print(regressor.intercept_)
#For retrieving the slope:
print(regressor.coef_)


# In[65]:


y_pred = regressor.predict(X_test)


# In[62]:


df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df


# In[ ]:




