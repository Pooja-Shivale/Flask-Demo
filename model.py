
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle


# In[2]:

df = pd.read_csv('/home/pooja/Downloads/Jupyter/LR-WW2/MLR-RedWine/winequality-red.csv')
df


# In[3]:

df.isnull().any()


# In[4]:

df = df.fillna(method='ffill') #forward fill ,used to fill preceding values


# In[5]:

x= df[['fixed acidity', 'volatile acidity', 'citric acid',
       'residual sugar', 'chlorides', 'free sulfur dioxide',
       'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']].values
y= df['quality'].values


# In[6]:

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# In[7]:

#training of model
lr= LinearRegression()
lr.fit(x_train,y_train)


# In[9]:

#saving model to disk
pickle.dump(lr, open('model.pkl','wb'))


# In[10]:

#loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[7.4,0.700 ,0.00 ,1.9 ,0.076,11.0 ,34.0 ,0.99780 ,3.51 ,0.56 ,9.4]]))


# In[ ]:



