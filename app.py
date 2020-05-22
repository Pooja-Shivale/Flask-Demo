#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np 
from flask import Flask, request, jsonify, render_template
import pickle


# In[7]:


app= Flask(__name__)
model= pickle.load(open('model.pkl','rb'))


# In[8]:


@app.route('/')
def home():
    return render_template('index1.html')


# In[ ]:


@app.route('/predict',methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features= [np.array(int_features)]
    prediction = model.predict(final_features)
    
    output = round(prediction[0],2)
    
    return render_template('index1.html', prediction_text='Quality of wine should be {}'.format(output))

if __name__== "__main__":
    app.run(debug=True)


# In[ ]:




# In[ ]:





# In[ ]:




