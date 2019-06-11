#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ShenSiyuan
"""

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
from keras.layers import Activation, Dense
from sklearn.preprocessing import MinMaxScaler
from keras.layers import TimeDistributed, RepeatVector
import random


# transform the features 
def pre_processing(filename):
    data = pd.read_csv(filename)
    
    data_sorted = data.sort_values(by=['DayOfWeek', 'Time'])
    data_sorted = data_sorted.reset_index(drop=True)
    window_size = 100
    times_transformed = []
    motion_count_list = [] # count how long the motion status is active
    acc_count_list = []
    temp_count_list = []
    temp_avg_list = []
    
    Motion = data_sorted["Motion"]
    Time = data_sorted["Time"]
    Acceleration = data_sorted["Acceleration"]
    Temp = data_sorted["Temperature"]
    
    for i in range(window_size-1, data_sorted.shape[0]):
        #time index 
        timestamp = Time[i]
        time = timestamp.split(':')
        h = (int)(time[0]) * 60 * 60
        m = (int)(time[1]) * 60
        s = (float)(time[2])
        time = h + m + s
        times_transformed.append((int)(time / 1800))
       
        #motion count 
        motion_count = 0
        for j in range(0, window_size):
            motion = Motion[i-j]
            if (motion == "active"):
                motion_count = motion_count + 1
        motion_count_list.append(motion_count)
                
        #acceleration count
        acc_count = 0
        for j in range(0, window_size):
            acc = Acceleration[i-j]
            if (acc == 'active'):
                acc_count = acc_count + 1
        acc_count_list.append(acc_count)
        
        #temperature change count
        temp_count = 0
        for j in range(0, window_size-1):
            if (Temp[i-j] != Temp[i-j-1]):
                temp_count = temp_count + 1
        temp_count_list.append(temp_count)
            
        #average temperature
        temp_sum = 0
        for j in range(0, window_size):
            temp_sum = temp_sum + Temp[i-j]
        temp_avg_list.append(temp_sum / window_size)
            
    data_input = np.array([times_transformed, motion_count_list,
                                 acc_count_list, temp_count_list, temp_avg_list])
    data_input =  data_input.transpose()
    
    #standardize the data with StandardScaler
    # using MinMax Scaler will cause the loss to be nan 
    scaler = StandardScaler()
    data_input = scaler.fit_transform(data_input)
    
    #Convert the input to the format required by LSTM
    X = []
    Y_labels = []
    
    for i in range(10 - 1, data_input.shape[0] - 1 ):
        temp = []
        for j in range(10):
            temp.append(data_input[i-j])
        X.append(temp)
        Y_labels.append(data_input[i+1])

    X = np.asarray(X)
    Y_labels = np.asarray(Y_labels)
    # Y_labels = np.reshape(Y_labels, (Y_labels.shape[0], 1, 5))

    return X, Y_labels


def train_predict(X, Y_labels):
    X_train, X_test, y_train, y_test = train_test_split(X, Y_labels, test_size=0.3, random_state=0)
    model = Sequential()
    model.add(LSTM(35, input_shape=(50, 5)))
    model.add(Dense(1, activation = 'tanh'))
    model.compile(loss='mean_squared_error', optimizer='adam',metrics = ['accuracy'])
    print(model.summary())

    history = model.fit(X_train, y_train, epochs=5, batch_size=10, verbose=2, shuffle=False)
    history
    y_pred = model.predict(X_test, verbose=0)
    return y_pred
    
    

def train_predict2(X, Y_labels):
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, Y_labels, test_size=0.2, random_state=0)
    # validation set
    model = Sequential()
    model.add(LSTM(35, input_shape=(50, 5)))
    model.add(Dense(1, activation = 'softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
    print(model.summary())

    history = model.fit(X_train, y_train, epochs=5, batch_size=5, verbose=2, shuffle=False)
    history
    y_pred = model.predict(X_test, verbose=0)
    print(classification_report(y_test, y_pred))
    return y_pred
    



'''
X, Y_labels = pre_processing()
y_pred = train_predict(X, Y_labels)
X = np.concatenate((X_norm, X_abnormal), axis=0)
Y = Y_norm + Y_abnormal
'''


#### deal with imlanance of classes
def overSampling(X_norm, X_abnormal):
    size_norm = X_norm.shape[0]
    size_abnormal = X_abnormal.shape[0]
    size_abnormal_original = size_abnormal
    while (size_abnormal <= size_norm):
        rand_number = random.randint(0, size_abnormal_original)
        sample_temp = X_abnormal[rand_number]
        sample_temp =  np.reshape(sample_temp, (1,50,5))
        X_abnormal = np.append(X_abnormal, sample_temp, axis = 0)
        size_abnormal = size_abnormal + 1
    Y_norm = [1] * X_norm.shape[0]
    Y_abnormal = [0] * X_abnormal.shape[0]
    return X_norm, X_abnormal, Y_norm, Y_abnormal




'''
split the normal data into training and validation sets, 
and the abnormal data as testing set
'''
X_norm, Y_norm = pre_processing("normal_data.csv")
# validation set 
X_abnormal, Y_abnormal = pre_processing("abnormal_data.csv")
### train validation split
X_train, X_vali, y_train, y_vali = train_test_split(X_norm, Y_norm, test_size=0.2, random_state=0)



model = Sequential()
model.add(LSTM(128, activation='relu', return_sequences=True, input_shape=(10,5)))
model.add(LSTM(128, activation='relu'))
model.add(Dense(5))

print(model.summary())
model.compile(loss='mse', optimizer='adam')
history = model.fit(X_train, y_train, epochs=5, validation_data=(X_vali, y_vali), batch_size=5, verbose=1, shuffle=True)
history
y_vali_pred = model.predict(X_vali, verbose=0)

'''
Layer (type)                 Output Shape              Param #   
=================================================================
lstm_33 (LSTM)               (None, 10, 128)           68608     
_________________________________________________________________
lstm_34 (LSTM)               (None, 128)               131584    
_________________________________________________________________
dense_13 (Dense)             (None, 5)                 645       
=================================================================
Total params: 200,837
Trainable params: 200,837
Non-trainable params: 0
_________________________________________________________________
None
Train on 79912 samples, validate on 19979 samples
Epoch 1/5
79912/79912 [==============================] - 353s 4ms/step - loss: 0.0274 - val_loss: 0.0153
Epoch 2/5
79912/79912 [==============================] - 350s 4ms/step - loss: 0.0139 - val_loss: 0.0138
Epoch 3/5
79912/79912 [==============================] - 350s 4ms/step - loss: 0.0131 - val_loss: 0.0133
Epoch 4/5
79912/79912 [==============================] - 350s 4ms/step - loss: 0.0127 - val_loss: 0.0127
Epoch 5/5
79912/79912 [==============================] - 353s 4ms/step - loss: 0.0125 - val_loss: 0.0124

'''


## Plot the graph of training and validation loss
plt.plot(history.history['loss'], linewidth=2, label='Train')
plt.plot(history.history['val_loss'], linewidth=2, label='Validation')
plt.legend(loc='upper right')
plt.show()


#### using mean square error to set a threshold for testing
#### the percentile is set to 40 here 
mse_list = []
for i in range(len(y_vali_pred)):
  mse_list.append(np.mean(np.square(np.subtract(y_vali_pred[i], y_vali[i]))))
threshold = np.percentile(mse_list, 95)
print("threshold is: ", threshold)

X_abnormal, y_abnormal = pre_processing("abnormal_data.csv")


t = 0
abnormal_mse = []
y_abnormal_pred = model.predict(X_abnormal)
pred_binary = []
for i in range(len(y_abnormal)):
  mse = np.mean(np.square(np.subtract(y_abnormal_pred[i], y_abnormal[i])))
  if mse < threshold:
    t = t + 1
    pred_binary.append(1)
  else:
    pred_binary.append(0)
  abnormal_mse.append(mse)
      
print("number of samples predicted to be normal in abnormal data: ", t)
print("number of abnormal data: ", len(y_abnormal))
print("abnormality accuracy():", (len(y_abnormal) - t)/ len(y_abnormal))




'''
Suppose positive = abnormal   negative = normal 

When the threshold percentile is set to 95 (false positive = 0.05), 
the abnormality accuracy (true positive) is 0.1800
count of times predicted to be normal in abnormal data:  8110
number of abnormal data:  9891
abnormality accuracy(): 0.18006268324739663



when set the threshold percentile to be 40 (false positive =  0.6), 
we get abnormality accuracy (true positive) of 0.6711151551
number of samples predicted to be normal in abnormal data:  3253
number of abnormal data:  9891
abnormality accuracy(): 0.6711151551915883
'''



'''
generate the ROC curve
'''
true_positive_rate = []
false_positive_rate = []
thresholds = []
for i in range(0, 100, 5):
  threshold = np.percentile(mse_list, i)
  t = 0
  for mse in abnormal_mse :
    if mse < threshold:
      t = t + 1
          
  true_positive_rate.append((len(y_abnormal) - t) / len(y_abnormal))
  false_positive_rate.append((100- i)/100)
  thresholds.append(threshold)
  


print(true_positive_rate)
print(false_positive_rate)

plt.plot(false_positive_rate, true_positive_rate)
plt.xlabel("false_positive_rate")
plt.ylabel("true_positive_rate")
plt.show()
    

