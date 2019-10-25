#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 10:58:02 2019

@author: ShenSiyuan
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from scipy.stats import chisquare
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
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




def pre_processing(filename, window_size = 20):
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
    
    times_transformed = []
    motion_count_list = [] # count how long the motion status is active
    acc_count_list = []
    temp_count_list = []
    temp_avg_list = []
    
    Motion = data_sorted["Motion"]
    Acceleration = data_sorted["Acceleration"]
    Temp = data_sorted["Temperature"]
    
    for i in range(window_size-1, data_sorted.shape[0]):

        #motion count 
        motion_count = 0
        for j in range(0, window_size):
            motion = Motion[i-j]
            if (motion == 1):
                motion_count = motion_count + 1
        motion_count_list.append(motion_count)
                
        #acceleration count
        acc_count = 0
        for j in range(0, window_size):
            acc = Acceleration[i-j]
            if (acc == 1):
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
    data_sorted_2 = data_sorted.iloc[window_size-1: data_sorted.shape[0], :]
    data_sorted_2 = data_sorted_2.reset_index(drop=True)
    data_sorted_2["accelerationCounts"] = acc_count_list
    data_sorted_2["motionCounts"] = motion_count_list
    data_sorted_2["averageTemperature"] = temp_avg_list
    data_sorted_2["temperatureCounts"] = temp_count_list
    return data_sorted_2



def pre_processing2(filename, timestep = 10):
    data = pd.read_csv(filename)
    
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


# apply exponentially weighted moving average

def pca_transform(data):
    scaler = MinMaxScaler()
    data[["accelerationCounts", "motionCounts", "temperatureCounts"]] = scaler.fit_transform(data[["accelerationCounts", "motionCounts", "temperatureCounts"]])

    pca = PCA(4)
    pca.fit(data)
    return pca.transform(data)
    

def find_correlation(data_normal):
    # calculate the pearson correlation between acceleration and acceleration count
    #0.24173811153354899
    corr, _ = pearsonr(data_normal["Acceleration"], data_normal["accelerationCounts"])
    
    #pearson correlation between motion and motion counts
    # 0.26798106937166738
    corr, _ = pearsonr(data_normal["Motion"], data_normal["motionCounts"])
    
    # pearson correlation between motion counts and acceleration counts
    # 0.6547714579986893   relatively high correlation 
    corr, _ = pearsonr(data_normal["accelerationCounts"], data_normal["motionCounts"])
    
    # 90 % of samples have the same result for motion and acceleration sensor
    count = 0
    for i in range(len(data_normal)):
        if (data_normal["Motion"][i] == data_normal["Acceleration"][i]):
            count = count + 1
    print(count/ len(data_normal))
    return 0 
    

def fingerprint(data_normal):
    sum_of_squared_distance = []
    num_clusters = []
    for i in range(2, 8):
        num_clusters.append(i)
        kmeans = KMeans(n_clusters=i, random_state=0).fit(data_normal)
        centroids = kmeans.cluster_centers_
        sum_of_squared_distance.append(kmeans.inertia_)
    plt.plot(num_clusters, sum_of_squared_distance)
    # set the number of clusters to be 4 
    kmeans = KMeans(n_clusters=4, random_state=0).fit(data_normal)
    kmeans_labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    return kmeans_labels, centroids
        
    

    
    
   
    
    

data_sorted = import_csv("normal_data.csv")
filename = "normal_data.csv"

data_normal = pre_processing("normal_data.csv", window_size = 20)
data_abnormal = pre_processing("abnormal_data.csv")

#data_normal_pca = pca_transform(data_normal)
#data_abnormal_pca = PCA(data_abnormal)

kmeans_labels, centroids = fingerprint(data_normal)


