#!/usr/bin/env python
# coding: utf-8

# # Task-4

# # Email Spam Detection with ML
# 
# 
# We've all been the recipient of spam emails before. Spam mail,or junk mail,is a type of mail that is sent to a massive number
# of users at one tome,frequently containing cryptic messages,scams,or most danngerously,phishing content.
# 
# In this project,use Python to build an email spam detector.Then,use machine learning to train the spam detector
# to recognize and classify emails and non-spam.

# In[26]:


#Importing the dependencies

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer  #converting the text data i.e., mail data into numerical values.So that our machine learning model can understand it easily.
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 


# # Data Collection and Pre-Processing

# In[27]:


#loading the data from csv file to a pandas dataframe
raw_mail_data=pd.read_csv("spam.csv",encoding='latin-1')


# In[35]:


raw_mail_data.head()


# In[36]:


print(raw_mail_data)


# In[37]:


#reppace all null mail values with a null string

mail_data=raw_mail_data.where((pd.notnull(raw_mail_data)),'')


# In[38]:


mail_data.head()


# In[39]:


mail_data.drop(['Unnamed: 2','Unnamed: 3', 'Unnamed: 4'], axis = 1, inplace = True)


# In[40]:


mail_data.head()


# In[41]:


mail_data.shape


# # Label Encoding

# In[60]:


#label spam mail as 0; ham mail as 1;
mail_data.loc[mail_data['v1']=='spam','v1',]=0
mail_data.loc[mail_data['v1']=='ham','v1',]=1


# spam-0
# ham-1

# In[61]:


#separating the data as texts and labels

x=mail_data['v2']
y=mail_data['v1']


# In[62]:


print(x)


# In[63]:


print(y)


# # Splitting the data into Train data and Test data

# In[64]:


X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# In[65]:


print(x.shape)
print(X_train.shape)
print(X_test.shape)


# # Feature Extraction

# In[66]:


#transform the text data to feature vectors that can be used as input to the Logistic Regression

feature_extraction=TfidfVectorizer(min_df=1,stop_words='english',lowercase=True)

X_train_features=feature_extraction.fit_transform(X_train)
X_test_features=feature_extraction.transform(X_test)

# Convert Y_train and Y_test into integers

Y_train=Y_train.astype('int')
Y_test=Y_test.astype('int')


# In[67]:


print(X_train_features)


# # Logistic Regression

# In[68]:


model=LogisticRegression()


# In[69]:


#training the logistic regression model with the training data

model.fit(X_train_features,Y_train)


# # Evaluting the Trained Model

# In[71]:


#prediction on training data

prediction_on_training_data=model.predict(X_train_features)
accuracy_on_training_data=accuracy_score(Y_train,prediction_on_training_data)


# In[72]:


print("Accuracy on training data:",accuracy_on_training_data)


# In[73]:


#prediction on test data

prediction_on_test_data=model.predict(X_test_features)
accuracy_on_test_data=accuracy_score(Y_test,prediction_on_test_data)


# In[74]:


print("Accuracy on test data:",accuracy_on_test_data)


# # Building a Predictive system

# In[90]:


input_mail=["URGENT! You have won a 1 week FREE membership in our å£100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net LCCLTD POBOX 4403LDNW1A7RW18"]

#convert text to feature extraction
input_data_features=feature_extraction.transform(input_mail)

#making predictions
prediction=model.predict(input_data_features)
print(prediction)


if prediction[0]==1:
    print("Ham mail")
else:
    print("spam mail")


# In[ ]:




