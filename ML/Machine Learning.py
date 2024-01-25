#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns



from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[11]:


df = pd.read_csv('emails.csv.zip', \
                 sep=',', encoding='latin-1')


# In[12]:


df.head()


# In[7]:


df = df.drop(["the", "to", "ect"], axis=1)
df = df.rename(columns={"v1":"label", "v2":"text"})
df.head()


# In[13]:


df.shape


# In[15]:


df.info()


# In[16]:


df.isnull().sum()


# In[17]:


df.drop('Email No.', axis=1, inplace=True)


# In[18]:


sns.countplot(data=df, x='Prediction')
plt.show()


# In[19]:


x = df.drop('Prediction', axis=1)
y = df['Prediction']


# In[20]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)


# In[21]:


model = LogisticRegression()
model.fit(x_train, y_train)


# In[22]:


y_pred = model.predict(x_test)


# In[23]:


def eval(name, y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    t1 = ConfusionMatrixDisplay(cm)
    print('Classification Report for Logistic Regression: \n')
    print(classification_report(y_test, y_pred))
    t1.plot()


# In[24]:


eval('Model Results', y_test, y_pred)


# In[ ]:




