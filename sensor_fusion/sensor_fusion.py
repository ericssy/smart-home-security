#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 10:58:02 2019

@author: ShenSiyuan
"""

import pandas as pd
import numpy as np


# Transform the features
def import_csv(filename):
    data = pd.read_csv(filename)
    time_transformed = []
    for i in range(data.shape[0]):
        timestamp = data["Time"][i]
        time = timestamp.split(':')
        h = (int)(time[0]) * 60 * 60
        m = (int)(time[1]) * 60
        s = (float)(time[2])
        time = (h + m + s)
        time_transformed.append(time)
    data["time_transformed"] = time_transformed
    data_sorted = data.sort_values(by=['DayOfWeek', 'time_transformed'])
    data_sorted = data_sorted.reset_index(drop=True)
    return data_sorted

def pre_processing(filename, timestep = 10):
    data = pd.read_csv(filename)
    data['Motion'] = data['Motion'].map({'inactive': 0, 'active': 1})
    data['Acceleration'] = data['Acceleration'].map({'inactive': 0, 'active': 1})
    data['DayOfWeek'] = data['DayOfWeek'] / 6.0
    time_transformed = []
    for i in range(data.shape[0]):
        timestamp = data["Time"][i]
        time = timestamp.split(':')
        h = (int)(time[0]) * 60 * 60
        m = (int)(time[1]) * 60
        s = (float)(time[2])
        time = (h + m + s)/ 86400
        time_transformed.append(time)
        
    data["time_transformed"] = time_transformed
    data = data.drop(["Time"], axis = 1)
    data["Temperature"] = (data["Temperature"] - data["Temperature"].min()) / (data["Temperature"].max() - data["Temperature"].min())
    data_sorted = data.sort_values(by=['DayOfWeek', 'time_transformed'])
    data_sorted = data_sorted.reset_index(drop=True)
    data_input = np.asarray(data_sorted)
    X = []
    Y_labels = []
    
    for i in range(timestep - 1, data_input.shape[0] - 1 ):
        temp = []
        for j in range(timestep):
            temp.append(data_input[i-j])
        X.append(temp)
        Y_labels.append(data_input[i+1])
    X = np.asarray(X)
    Y_labels = np.asarray(Y_labels)
    return X, Y_labels

def pre_processing2(filename, timestep = 10):
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
    
    
    for i in range(timestep - 1, data_input.shape[0] - 1 ):
        temp = []
        for j in range(timestep):
            temp.append(data_input[i-j])
        X.append(temp)
        Y_labels.append(data_input[i+1])

    X = np.asarray(X)
    Y_labels = np.asarray(Y_labels)
    # Y_labels = np.reshape(Y_labels, (Y_labels.shape[0], 1, 5))

    return X, Y_labels

