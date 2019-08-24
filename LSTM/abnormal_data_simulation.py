#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 21:11:17 2019

@author: ericshen

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import stats
import seaborn as sns, numpy as np
from datetime import datetime

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



def add_time_transformed(data):
    time_transformed = []
    for i in range(data.shape[0]):
        timestamp = data["Time"][i]
        time = timestamp.split(':')
        h = (int)(time[0]) * 60 * 60
        m = (int)(time[1]) * 60
        s = (float)(time[2])
        time = (h + m + s)
        time_transformed.append(time)
    new_data = data
    new_data["time_transformed"] = time_transformed
    return new_data



def simulate_normal_temperature():
    data_simulated = data_abnormal
    ## time series data 
    temp_simulated = scipy.stats.beta.rvs(200.2, 197.589, loc=11.907, scale=120.4149, size=data_simulated.shape[0], random_state=None)
    temp_simulated = temp_simulated.astype(int)
    data_simulated["Temperature"] = temp_simulated 
    data_simulated = data_simulated.drop(columns=["time_transformed"])
    return data_simulated
    


def average_time_difference(data):
    time_difference = []
    for Index, row in data.iterrows():
        if (Index == 0):
            time_transformed_previous = row["time_transformed"]
            day_of_week_previous = row["DayOfWeek"]
            continue
        if (day_of_week_previous == row["DayOfWeek"]):
            time_difference.append(row["time_transformed"] - time_transformed_previous)
        time_transformed_previous = row["time_transformed"]
        day_of_week_previous = row["DayOfWeek"]
    mean = sum(time_difference) / len(time_difference)
    ax = sns.distplot(time_difference)
    return time_difference, mean
        
       

def average_time_difference_at_least_one_active(data):
    '''
    Find out under normal situation, once there are at least one active events, what 
    the average time difference between this detection and the next detection is 
    '''
    time_difference = []
    for Index, row in data.iterrows():
        if (Index == data.shape[0] - 1):
            break
        if (row["DayOfWeek"] == data["DayOfWeek"][Index + 1]):
            if (row["Motion"] == "active" or row["Acceleration"] == "active"):
                time_difference.append(data["time_transformed"][Index + 1] - row["time_transformed"] )
    mean = sum(time_difference) / len(time_difference)
    return time_difference, mean
                


def average_time_difference_no_active(data):
    '''
    Find out the average time difference between two detections in the normal
    data, when there is no single active status in the first detection. The status
    in the second detection does not matter
    '''
    time_difference = []
    for Index, row in data.iterrows():
        if (Index == data.shape[0] - 1):
            break
        if (row["DayOfWeek"] == data["DayOfWeek"][Index + 1]):
            if (row["Motion"] == "inactive" and row["Acceleration"] == "inactive"):
                time_difference.append(data["time_transformed"][Index + 1] - row["time_transformed"] )
    mean = sum(time_difference) / len(time_difference)
    return time_difference, mean
    

def detection_with_at_least_one_inactive_distribution(data):
    '''
    Find the distribution of the time lag between two nearest detections in the
    abnormal data with at 
    least one 'inactive' status
    '''
    previous_at_least_one_active = 0
    previous_dayofweek = 0
    time_lag = []
    df = pd.DataFrame()
    dict1 = {}
    for Index, row in data.iterrows():
        if (row["Motion"] == "inactive" or row["Acceleration"] == "inactive"):
            df = df.append(row.to_frame().transpose())
    for Index, row in df.iterrows():
        if (previous_dayofweek == row["DayOfWeek"]):
            time_lag.append(row["time_transformed"] - previous_at_least_one_active)
            dict1[previous_at_least_one_active, row["time_transformed"] ] = row["time_transformed"] - previous_at_least_one_active
        previous_dayofweek = row["DayOfWeek"]
        previous_at_least_one_active = row["time_transformed"]
    mean = sum(time_lag) / len(time_lag)
    return time_lag, mean

            

def detection_lag_with_at_least_one_inactive_distribution(data):
    '''
    Find the distribution of the number of detection lag between two nearest detections in the
    abnormal data with at 
    least one 'inactive' status
    '''
    previous_at_least_one_active = 0
    previous_dayofweek = None
    lag = []
    for Index, row in data.iterrows():
        if (previous_at_least_one_active == 0 and (row["Motion"] == "inactive" or row["Acceleration"] == "inactive")):
            previous_at_least_one_active = Index
            previous_dayofweek = row["DayOfWeek"]
            continue
        if (row["Motion"] == "inactive" or row["Acceleration"] == "inactive" and (row["DayOfWeek"] == previous_dayofweek)):
            lag.append(Index - previous_at_least_one_active)
            previous_at_least_one_active = Index
            previous_dayofweek = row["DayOfWeek"]
    mean = sum(lag) / len(lag)
    ax = sns.distplot(lag)
    return lag, mean

  

# generate 10000 abnormal samples
def simulate_abnormal_data():
    # first simulate time 
    dayofweek = 0
    next_at_least_one_inactive = scipy.stats.beta.rvs(0.57083732450306168, 33.300996483902153, loc=0.019000000000232799, scale=9424.6788246838551, size=1, random_state=None)[0]
    time_sum = 0 
    time = 0
    time_previous = 0
    Time = []
    DayOfWeek = []
    Motion = []
    Acceleration = []
    Temperature = []
    for i in range(100000):
        # distribution of detection time difference 
        time_difference = scipy.stats.beta.rvs(1.3941664339418001, 95472646977931.203, loc=-0.068431330482272817, scale=365982486365332.12, size=1, random_state=None)[0]
        time = time_previous + time_difference  
        if (time < 60 * 60 * 24):
            m, s = divmod(time, 60)
            h, m = divmod(m, 60)
            s = "{:.3f}".format(s)
            timestamp = f'{int(h):02d}:{int(m):02d}:{s}'
            Time.append(timestamp)
            time_previous = time
            DayOfWeek.append(dayofweek)                
        else:
            time_previous = time_difference
            m, s = divmod(time_difference, 60)
            h, m = divmod(m, 60)
            s = "{:.3f}".format(s)
            timestamp = f'{int(h):02d}:{int(m):02d}:{s}'
            Time.append(timestamp)
            dayofweek = dayofweek + 1
            DayOfWeek.append(dayofweek)
        if (time_sum > next_at_least_one_inactive):
            next_at_least_one_inactive = scipy.stats.beta.rvs(0.57083732450306168, 33.300996483902153, loc=0.0028500000000347614, scale=1413.7041856079099, size=100, random_state=None)[0]
            time_sum = 0
            choice = np.random.choice(["s1", "s2", "s3"] , 1, p=[0.5, 0.25, 0.25])[0]
            if (choice == "s1"):
                Motion.append("inactive")
                Acceleration.append("inactive")
            if (choice == "s2"):
                Motion.append("active")
                Acceleration.append("inactive")
            if (choice == "s3"):
                Motion.append("inactive")
                Acceleration.append("active")      
        else:
            time_sum = time_sum + time_difference  
            Motion.append("active")
            Acceleration.append("active")
        Temperature.append(scipy.stats.beta.rvs(116.80683031739741, 153.47345646038463, loc=28.454520874016371, scale=120.4149, size=1, random_state=None)[0])
    simulated_df = pd.DataFrame()
    simulated_df["Time"] = Time
    simulated_df["DayOfWeek"] = DayOfWeek
    simulated_df["Motion"] = Motion
    simulated_df["Acceleration"] = Acceleration
    simulated_df["Temperature"] = Temperature
    
    simulated_df.to_csv("simulated_data.csv")
    return simulated_df

    
###########################################
    

data_norm = import_csv("normal_data.csv")
data_abnormal = import_csv("abnormal_data.csv")

abnormal_with_normal_temp = simulate_normal_temperature()
abnormal_with_normal_temp.to_csv("abnormal_with_simulated_normal_temp.csv")
#data_normal_temp_only["Temperature"].describe()


### average of time difference for abnormal is 43.0406 seconds, for normal is 6.04
### The time internval for abnormal should be at least as short as that for normal data, 
### because in the normal data, once we have one or two 'active" events, the
### sensor would record the data of that room almost immediately, often within
### two seconds. 
time_difference_norm, avg_time_diff_norm = average_time_difference(data_norm)
time_difference_abnormal, avg_time_diff_abnormal = average_time_difference(data_abnormal)


### next, find out in normal situation, once there is at least one active events, what 
### is the average time difference between this detection and the next detection
### avg_time_diff_at_least_one_active is 6.035. There is no significant time difference
### in these two situations. 
time_diff_norm_at_least_one_active, avg_time_diff_at_least_one_active = average_time_difference_at_least_one_active(data_norm)
print(avg_time_diff_at_least_one_active)
### The average time difference for detection with no active status has an
### average of 6.05004322182. It further proves that the acceleration and
### motion status have no influence on how soon the next detection will occur
time_diff_norm_no_active, avg_time_diff_no_active = average_time_difference_no_active(data_norm)
print(avg_time_diff_no_active)


time_lag, mean = detection_with_at_least_one_inactive_distribution(data_abnormal)
print(mean)
### trying to fit a beta distribution
time_lag_for_simulation = [x / 20 for x in time_lag]
scipy.stats.beta.fit(time_lag_for_simulation)
### output: (0.57083732450306168, 33.300996483902153, 0.0028500000000347614, 1413.7041856079099)
scipy.stats.beta.rvs(0.57083732450306168, 33.300996483902153, loc=0.0028500000000347614, scale=1413.7041856079099, size=100, random_state=None)

### for simultion, fit a 
lag, mean_num_lag = detection_lag_with_at_least_one_inactive_distribution(data_abnormal)
print(mean_num_lag)
scipy.stats.beta.fit(mean_num_lag)


data_abnormal.iloc[8990]

### fit a beta distribution on time_difference_norm  (the time difference 
### of two consecutive detections) for each sample
scipy.stats.beta.fit(time_difference_norm)
### (1.3941664339418001, 95472646977931.203, -0.068431330482272817, 365982486365332.12)
scipy.stats.beta.rvs(1.3941664339418001, 95472646977931.203, loc=-0.068431330482272817, scale=365982486365332.12, size=100, random_state=None)


## distribution of temperature 
data_norm["Temperature"].describe()
data_abnormal["Temperature"].describe()
ax = sns.distplot(data_abnormal["Temperature"])
plt.show()

temp_difference = []
for i in range(1, data_norm.shape[0]):
    temp_difference.append(data_norm.iloc[i]["Temperature"] - data_norm.iloc[i-1]["Temperature"])
temp_difference = np.asarray(temp_difference)
stats.describe(temp_difference)

ax = sns.distplot(data_norm["Temperature"])
plt.show()
scipy.stats.beta.fit(data_norm["Temperature"])
scipy.stats.beta.fit(data_abnormal["Temperature"])
scipy.stats.beta.rvs(200.2, 197.589, loc=11.907, scale=120.4149, size=100, random_state=None)



# plot the distribution of temperature difference 
ax = sns.distplot(temp_difference)

# looks like it's a normal distribution. As the temperature difference distribution has a
# hard upper and lower limits, we use beta distribution. 
scipy.stats.beta.fit(temp_difference) 
scipy.stats.beta.rvs(128.9, 138.9, loc=-67.36, scale=139.96, size=100, random_state=None)


############# simulate abnormal data and save the data as a csv file ##################
simulated_df = simulate_abnormal_data()


### find the average time difference between two detection in simulated_df
temp1 = add_time_transformed(simulated_df)
_, avg_simulated = average_time_difference(simulated_df)
print(avg_simulated)

