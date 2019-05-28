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

def read_data():
    data = pd.read_csv("normal_data.csv")
    return data
    



def pre_processing(data):
     #Sort the data by time
     # first by DayOfWeek, then by Time
    data_sorted = data.sort_values(by=['DayOfWeek', 'Time'])
    data_input = data_sorted.drop(['Time', 'DayOfWeek'], axis = 1)
    le = preprocessing.LabelEncoder()
    data_input = data_input.apply(le.fit_transform)
    data_input = data_input.reset_index(drop=True)
    
    ct = ColumnTransformer([
        ('somename', StandardScaler(), ['Temperature'])],
        remainder='passthrough')
    data_input_2 = ct.fit_transform(data_input)
    X = []
    for i in range(49, 9999):
        temp = []
        for j in range(50):
            temp.append(data_input_2[i-j])
        X.append(temp)
    X = np.asarray(X)
    Y_labels = [1] * X.shape[0]
    
    return X, Y_labels
    

# transform the features 
def pre_processing2(data):
    data_sorted = data.sort_values(by=['DayOfWeek', 'Time'])
    data_sorted = data_sorted.reset_index(drop=True)
    window_size = 100
    times_transformed = []
    motion_count_list = [] # count how long the motion status is active
    acc_count_list = []
    temp_count_list = []
    temp_sum_list = []
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
    
    #standardize the data with MinMaxScaler
    scaler = MinMaxScaler()
    data_input = scaler.fit_transform(data_input)
    
    #Convert the input to the format required by LSTM
    X = []
    for i in range(50 - 1, data_input.shape[0] ):
        temp = []
        for j in range(50):
            temp.append(data_input[i-j])
        X.append(temp)
    X = np.asarray(X)
    Y_labels = [1] * X.shape[0]
    
    return 0


'''
# ### Train the LSTM model 
# #### add LSTM with input shape (50,3): 50 time steps and 3 features

model = Sequential()
model.add(LSTM(56, input_shape=(50, 3)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())
X_train, X_test, y_train, y_test = train_test_split(X, Y_labels, test_size=0.3, random_state=0)



X_train.shape, len(y_train)


history = model.fit(X_train, y_train, epochs=5, batch_size=10, verbose=2, shuffle=False)
history

accr = model.evaluate(X_test,y_test)
accr


prediction = model.predict(X_test, verbose=0)
prediction.tolist()


prediction.shape


y_test

len(y_test)


for i in range(len(y_test)):
    y_test[i] = 1

for i in range(len(y_train)):
    y_train[i] = 1 
    
y_train
'''



def train_predict(X, Y_labels):
    X_train, X_test, y_train, y_test = train_test_split(X, Y_labels, test_size=0.3, random_state=0)
    model = Sequential()
    model.add(LSTM(35, input_shape=(50, 3)))
    model.add(Dense(1, activation = 'tanh'))
    model.compile(loss='mean_squared_error', optimizer='adam',metrics = ['accuracy'])
    print(model.summary())

    history = model.fit(X_train, y_train, epochs=5, batch_size=10, verbose=2, shuffle=False)
    history
    y_pred = model.predict(X_test, verbose=0)
    return y_pred
    
    

def train_predict2(X, Y_labels):
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
    



#################
data = read_data()
X, Y_labels = pre_processing(data)
y_pred = train_predict(X, Y_labels)

X, Y_labels = pre_processing2(data)
y_pred_2 = train_predict2(X, Y_labels)
