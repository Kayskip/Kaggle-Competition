#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy
import pandas as pd
import numpy as np
import random as rnd

from matplotlib.pyplot import figure
# visualization
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# In[2]:


#Loading the dataset
data = pd.read_csv('/home/karu/Documents/COMP309/Assignment 3/core/Given Datasets/CoreDataSet-Train.csv',sep=',')


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data.isnull().sum()


# In[6]:


correlations = data.corr()


# In[19]:


names = ['id', 'difficulty', 'shape_length', 'X', 'Y', 'Class']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=1, vmax=0)
fig.colorbar(cax)
ticks = numpy.arange(0,5,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()


# In[ ]:




