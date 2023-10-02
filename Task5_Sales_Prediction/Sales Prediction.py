#!/usr/bin/env python
# coding: utf-8

# # Task-5

# # Sales Prediction Using Python
# 
# Sales prediction means predictiong how much of a product people will buy based 
# on factors such as the amount you spend to advertise your product,the segment
# of people you advertise for,or the platform you are advertising on abouy your
# product.
# 
# Typically,a product and servise-based business always need their Data Scientist
# to predict their future sales with every step they take to manipulate the cost
# of advertising their product.

# In[1]:


#Importing the important Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# # Data Collection and Pre-Processing

# In[2]:


data=pd.read_csv("Advertising.csv")
data.head()


# In[3]:


data.drop(['Unnamed: 0'],axis=1,inplace=True)


# In[4]:


data.head()


# In[5]:


data.shape


# In[6]:


data.describe()


# Basic Observation

# Avg expense spend is highest on TV
# 
# Avg expense spend is lowest on Radio
# 
# Max sale is 27 and min is 1.6

# In[8]:


#Pair plot

sns.pairplot(data,x_vars=['TV','Radio','Newspaper'],y_vars='Sales',kind='scatter')
plt.show()


# Pair Plot Observation
# 
# When Advertising cost increases in TV Ads the sales will increase as well.
# While the Newspaper and Radio is a bit Unpredictiable.

# In[9]:


plt.hist(data['TV'],bins=10)


# In[17]:


plt.hist(data['Newspaper'], bins=10, color='orange')
plt.xlabel('Newspaper')# Add this line to set the x-axis label
plt.ylabel('Frequency')
plt.show()


# In[18]:


plt.hist(data['Radio'], bins=10, color='green')
plt.xlabel('Radio')  # Add this line to set the x-axis label
plt.ylabel('Frequency')
plt.show()


# Histogram Observation
# 
# The majority sales is the result of low Advertising cost in newspaper

# In[19]:


data.corr()


# In[23]:


sns.heatmap(data.corr(),annot=True)
plt.show()


# SALES IS HIGHLY CORRELATED WITH TV

# # Linear Regession

# In[24]:


#Lets train our model using linear regression as it it is correlated with only variable TV

x=data['TV'] #input variable as it is higghly correlated with sales


# In[25]:


y=data["Sales"] #Target variable


# In[26]:


print(x)


# In[27]:


print(y)


# In[28]:


X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[34]:


#Building th emodel

# Assuming X_train is a 1D array
X_train_reshaped = X_train.values.reshape(-1, 1)

# Create the Linear Regression model
model = LinearRegression()

# Fit the model
model.fit(X_train_reshaped, Y_train)


# In[40]:


#Predicting the trained data
prediction_train=model.predict(X_train.values.reshape(-1,1))


# In[41]:


print(prediction_train)


# In[44]:


#predicting the test data
prediction_test=model.predict(X_test.values.reshape(-1,1))


# In[45]:


print(prediction_test)


# In[48]:


pre=model.predict(Y_test.values.reshape(-1,1))
print(pre)


# In[49]:


model.coef_


# In[50]:


model.intercept_


# In[54]:


0.04600779*60.2+7.2924937735593645


# In[55]:


plt.plot(prediction_test)


# In[56]:


plt.scatter(X_test,Y_test)
plt.plot(X_test,7.2924937735593645+0.04600779*X_test,'r')
plt.show()


# Conclusion

# So this is how we can predict future sales of a product with machine learning.

# In[ ]:




