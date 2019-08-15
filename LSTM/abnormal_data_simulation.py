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
    return time_difference, mean
        
        
    

# generate 100000 abnormal samples
# the average time difference for the original abnormal data is 8.6 secs, for normal
# data, it;s 0.86 secs, the data should be 
def simulate_abnormal_data():
    for i in range(100000):
        time = 
    
    
    



###########################################
    

data_norm = import_csv("normal_data.csv")
data_abnormal = import_csv("abnormal_data.csv")

abnormal_with_normal_temp = simulate_normal_temperature()
abnormal_with_normal_temp.to_csv("abnormal_with_simulated_normal_temp.csv")
#data_normal_temp_only["Temperature"].describe()




### average of time difference for abnormal is 8.6 seconds, for normal is 0.86
time_difference_norm, avg_time_diff_norm = average_time_difference(data_norm)
time_difference_abnormal, avg_time_diff_abnormal = average_time_difference(data_abnormal)
    

### once active, detect again with in 500 mili second



## distribution of temperature 
data_norm["Temperature"].describe()
data_abnormal["Temperature"].describe()
plt.plot(data_norm.index,data_norm["Temperature"])
plt.show()

temp_difference = []
for i in range(1, data_norm.shape[0]):
    temp_difference.append(data_norm.iloc[i]["Temperature"] - data_norm.iloc[i-1]["Temperature"])
temp_difference = np.asarray(temp_difference)
stats.describe(temp_difference)

ax = sns.distplot(data_norm["Temperature"])
scipy.stats.beta.fit(data_norm["Temperature"])
scipy.stats.beta.rvs(200.2, 197.589, loc=11.907, scale=120.4149, size=100, random_state=None)



# plot the distribution of temperature difference 
ax = sns.distplot(temp_difference)

# looks like it's a normal distribution. As the temperature difference distribution has a
# hard upper and lower limits, we use beta distribution. 
scipy.stats.beta.fit(temp_difference) 
scipy.stats.beta.rvs(128.9, 138.9, loc=-67.36, scale=139.96, size=100, random_state=None)



'''
data_norm["Temperature"].describe()
Out[44]: 
count    100000.000000
mean         72.511580
std           3.014859
min          59.000000
25%          70.000000
50%          73.000000
75%          75.000000
max          85.000000
Name: Temperature, dtype: float64

data_normal_temp_only["Temperature"].describe()
Out[106]: 
count    10000.000000
mean        72.023700
std          3.045204
min         60.000000
25%         70.000000
50%         72.000000
75%         74.000000
max         84.000000


stats.describe(temp_difference)
DescribeResult(nobs=99999, minmax=(-18, 20), 
mean=1.000010000100001e-05, variance=18.18893377857557, 
skewness=0.008871915571305131, kurtosis=-0.022606768908992603)
'''
    




    
    
    