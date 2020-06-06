#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import required libraries
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[2]:


# Reading the dataset of House Sales
df = pd.read_csv("G:\\Multicollinearity\\House Sales.csv")


# In[3]:


df.head()


# In[4]:


type(df)


# ### Calculating VIF scores for original data

# In[5]:


# Creating a function to calculate the VIF scores for all independant features with for loop


def vif_scores(df):
    VIF_Scores = pd.DataFrame()
    VIF_Scores["Independent Features"] = df.columns
    VIF_Scores["VIF Scores"] = [variance_inflation_factor(df.values,i) for i in range(df.shape[1])]
    return VIF_Scores

df1 = df.iloc[:,:-1]
vif_scores(df1)


# ### Fixing Multicollinearity - dropping variables

# In[6]:


#Copying the original dataframe
df2 = df.copy()


# In[7]:


# Dropping the features which are having high VIF values
df3 = df2.drop(['Interior(Sq Ft)','# of Rooms'], axis = 1)


# In[8]:


df3.head()


# In[9]:


#Calculating VIF scores after dropping the varaibles
def vif_scores(df3):
    VIF_Scores = pd.DataFrame()
    VIF_Scores["Independant Features"] = df3.columns
    VIF_Scores["VIF Scores"] = [variance_inflation_factor(df3.values,i) for i in range(df3.shape[1])]
    return VIF_Scores

df3 = df3.iloc[:,:-1]
vif_scores(df3)


# ### Fixing multicollinearity - Combining the variables

# In[10]:


df4= df3.copy()


# In[11]:


df4.head()


# In[12]:


#Combining the variables and calculating the VIF scores
df5 = df4.copy()
df5['Total Rooms'] = df4.apply(lambda x: x['# of Bed'] + x['# of Bath'],axis=1)
X = df5.drop(['# of Bed','# of Bath'],axis=1)
vif_scores(X)

