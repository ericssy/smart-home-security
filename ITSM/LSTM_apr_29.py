
# coding: utf-8

# In[35]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
from sklearn import preprocessing
import itertools 
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.model_selection import train_test_split
from sklearn import datasets


# In[36]:

data = pd.read_csv("normal_data.csv")
data.head()


# ### Sort the data by time
# #### first by DayOfWeek, then by Time

# In[37]:

data_sorted = data.sort_values(by=['DayOfWeek', 'Time'])
data_sorted


# In[38]:

(data_sorted[['Temperature']]).max()


# In[39]:

(data_sorted[['Temperature']]).min()


# ### 56 possible distinct log keys (14 * 4)

# In[40]:

data_sorted.groupby(['Motion', 'Acceleration', 'Temperature']).size()


# In[ ]:




# ### input data format for LSTM 
# 

# In[41]:

data_input = data_sorted.drop(['Time', 'DayOfWeek'], axis = 1)
data_input.head()


# In[42]:

le = preprocessing.LabelEncoder()
data_input = data_input.apply(le.fit_transform)
data_input


# ### assign the class for each sample of logs

# In[43]:

l=[ [0, 1], [0, 1] ]
temp = []
for i in range(1, 15):
    temp.append(i)
l.append(temp)

lookup_list = list(itertools.product(*l)) 
lookup_list


# In[44]:

data_input = data_input.reset_index(drop=True)
data_input


# In[ ]:




# In[45]:

category = []
for i in range(10000):
    category.append(0)
len(category )


# In[46]:


for index, row in data_input.iterrows():
    for i in range(56):
        if (lookup_list[i][0] == row['Motion'] and lookup_list[i][1] == row['Acceleration'] 
            and lookup_list[i][2] == row['Temperature']):
            category[index] = i
               


# In[ ]:




# In[47]:

category


# In[48]:


ct = ColumnTransformer([
        ('somename', StandardScaler(), ['Temperature'])
    ], remainder='passthrough')

data_input_2 = ct.fit_transform(data_input)
data_input_2[0:20]


# In[49]:

data_input_2.shape


# ### method II for labels

# In[ ]:




# ### assume 50 time steps

# In[74]:

Y_labels = []
X = []
for i in range(49, 9999):
    temp = []
    for j in range(50):
        temp.append(data_input_2[i-j])
    X.append(temp)
    Y_labels.append(category[i+1])
        
        
    


# In[75]:

X = np.asarray(X)


# In[52]:

X


# ### Check the size of the features and the labels

# In[76]:

X.shape


# In[77]:

len(Y_labels )


# ### Train the LSTM model 
# #### add LSTM with input shape (50,3): 50 time steps and 3 features

# In[78]:

model = Sequential()
model.add(LSTM(8, input_shape=(50, 3)))
model.add(Dense(1))
model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())


# In[79]:

X_train, X_test, y_train, y_test = train_test_split(X, Y_labels, test_size=0.3, random_state=0)


# In[80]:

X_train.shape, len(y_train)


# In[ ]:




# In[81]:

history = model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=2, shuffle=False)
history


# In[61]:

accr = model.evaluate(X_test,y_test)
accr

