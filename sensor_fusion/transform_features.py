
# coding: utf-8

# In[176]:

import pandas as pd
import numpy as np
import time
import datetime
pd.set_option('display.max_rows', 1000)
from scipy.stats import truncnorm
from scipy import signal
import matplotlib.pyplot as plt


# There are 3 sensors in total. Two old sensors and the Aeotect Multisensor 6 
# * sensorA: multipurpose sensor
#  * acceleration
#  * status (door)
#  * temperature 
# * sensorB: motion sensor
#  * motion
# * sensorC: multisensor 
#  * motion
#  * temperature
#  * humidity
#  * illuminance
#  

# In[109]:


sensorA = pd.read_csv("data_multipurpose_sensor.csv")
sensorB = pd.read_csv("data_motion_sensor.csv")
sensorC = pd.read_csv("data_multisensor.csv")


# In[ ]:




# In[110]:

sensorA 


# In[111]:

sensorB


# In[112]:

sensorC


# In[113]:

sensorA["name"] = sensorA["name"] + "_sensorA"


# In[114]:

sensorA["name"]


# In[115]:

sensorA


# In[ ]:




# In[116]:

sensorB["name"] = sensorB["name"] + "_sensorB"
sensorC["name"] = sensorC["name"] + "_sensorC"


# combine the three datasets 

# In[117]:

data = pd.concat([sensorA, sensorB, sensorC])


# In[118]:

data


# rank the data by time, with ascending order 
# first conver the time to timestamp

# In[119]:

data.iloc[2323]


# In[120]:


timestamp_list = []
for i in range(data.shape[0]):
    #epoch = int(time.mktime(datetime.datetime.strptime(data["time"].iloc[i],"%Y-%m-%d %I:%M:%S.%f %p")))
    timestamp = datetime.datetime.strptime(data["time"].iloc[i],"%Y-%m-%d %I:%M:%S.%f %p").timestamp()
    timestamp_list.append(timestamp)
timestamp_list


# In[121]:

data["timestamp"] = timestamp_list


# In[122]:

data = data.sort_values(by = ["timestamp"], ascending = True)
data = data.reset_index(drop = True)


# In[131]:

data = data.iloc[340:]


# In[132]:

data


# hard code to select data in this afternoon

# In[133]:

data.name.unique().tolist()


# In[134]:

column_list = data.name.unique().tolist()
try: 
    column_list.remove("tamper_sensorC")
except ValueError:
    print("")

try: 
    column_list.remove("battery_sensorA")
except ValueError:
     print("")


try: 
    column_list.remove("threeAxis_sensorA")
except ValueError:
     print("")
    

try: 
    column_list.remove("contact_sensorA")
except ValueError:
     print("")
        
try: 
    column_list.remove("ping_sensorA")
except ValueError:
     print("")
    
try: 
    column_list.remove("activity_sensorA")
except ValueError:
     print("")
        
try: 
    column_list.remove("battery_sensorB")
except ValueError:
     print("")

try: 
    column_list.remove("activity_sensorC")
except ValueError:
     print("")

try: 
    column_list.remove("ping_sensorC")
except ValueError:
     print("")        


# # The transformed dataset

# In[135]:

# initialize the values
df = pd.DataFrame(columns = ['temperature_sensorA', 'acceleration_sensorA', 'status_sensorA', 'motion_sensorB', 'motion_sensorC', 'temperature_sensorC', 'humidity_sensorC', 'illuminance_sensorC', 'time'] )
previous_dict = {'temperature_sensorB' : 70, 'temperature_sensorA' : 70, 'acceleration_sensorA' : "inactive", 'status_sensorA' : "closed", 'motion_sensorB':"inactive", 'motion_sensorC' : "inactive", 'temperature_sensorC' : 70, 'humidity_sensorC' : 62, 'illuminance_sensorC' : 4}
for index, row in data.iterrows():
    time =  row["time"]
    name = row["name"]
    value = row["value"]
    if (name in column_list):
        for item in column_list:
            df.at[index, item] = previous_dict[item]
        #df[name].iloc[index] = 
        df.at[index, name] = value
        previous_dict[name] = value
        df.at[index, "time"] = time
df


# In[138]:

df.to_csv("normal_multisensor.csv")


# In[137]:

def transform_data():
    sensorA = pd.read_csv("data_multipurpose_sensor.csv")
    sensorB = pd.read_csv("data_motion_sensor.csv")
    sensorC = pd.read_csv("data_multisensor.csv")
    sensorA["name"] = sensorA["name"] + "_sensorA"
    sensorB["name"] = sensorB["name"] + "_sensorB"
    sensorC["name"] = sensorC["name"] + "_sensorC"
    #combine the data
    data = pd.concat([sensorA, sensorB, sensorC])
    timestamp_list = []
    for i in range(data.shape[0]):
        #epoch = int(time.mktime(datetime.datetime.strptime(data["time"].iloc[i],"%Y-%m-%d %I:%M:%S.%f %p")))
        timestamp = datetime.datetime.strptime(data["time"].iloc[i],"%Y-%m-%d %I:%M:%S.%f %p").timestamp()
        timestamp_list.append(timestamp)
    data["timestamp"] = timestamp_list
    data = data.sort_values(by = ["timestamp"], ascending = True)
    data = data.reset_index(drop = True)
    data = data.iloc[340:]
    column_list = data.name.unique().tolist()
    column_list.remove("tamper_sensorC")
    column_list.remove("battery_sensorA")
    df = pd.DataFrame(columns = ['temperature_sensorA', 'acceleration_sensorA', 'status_sensorA', 'motion_sensorB', 'motion_sensorC', 'temperature_sensorC', 'humidity_sensorC', 'illuminance_sensorC', 'time'] )
    previous_dict = {'temperature_sensorA' : 80, 'acceleration_sensorA' : "inactive", 'status_sensorA' : "closed", 'motion_sensorB':"inactive", 'motion_sensorC' : "inactive", 'temperature_sensorC' : 72.5, 'humidity_sensorC' : 62, 'illuminance_sensorC' : 4}
    for index, row in data.iterrows():
        time =  row["time"]
        name = row["name"]
        value = row["value"]
        if (name in column_list):
            for item in column_list:
                df.at[index, item] = previous_dict[item]
            #df[name].iloc[index] = 
            df.at[index, name] = value
            previous_dict[name] = value
            df.at[index, "time"] = time


# In[139]:

df.shape[0]


# ## Simulate a set of temperatures for a potential attacker device.  The mean and variance in the simulated data are similar while the frequency and interval of temperature changes are different, since the attcker device does not know what's happeing in the room. 

# In[149]:

simulated_temp = []
for i in range(df.shape[0]):
    temp = int((truncnorm((65 - 75) / 2, (80 - 75) / 2, loc=75, scale=2).rvs()))
    simulated_temp.append(temp)
    


# In[158]:

simulated_temp


# ## compare the simulated temperature with the tempreature recored by the sensor 

# In[162]:

df["temperature_sensorA"].astype(int).to_numpy()


#  ## measure the cross correlation 

# In[163]:

signal.correlate(simulated_temp, df["temperature_sensorA"].astype(int).to_numpy(), mode='same') 


# In[165]:

signal.correlate(df["temperature_sensorB"].astype(int).to_numpy(), df["temperature_sensorA"].astype(int).to_numpy(), mode='same') 


# In[170]:

signal.correlate(df["temperature_sensorC"].astype(float).to_numpy(), df["temperature_sensorA"].astype(int).to_numpy(), mode='same').tolist()


# ##  time lagged cross correlation 

# In[172]:

def crosscorr(datax, datay, lag=0, wrap=False):
    """ Lag-N cross correlation. 
    Shifted data filled with NaNs 
    
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length
    Returns
    ----------
    crosscorr : float
    """
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else: 
        return datax.corr(datay.shift(lag))


# In[181]:

d1 = df["temperature_sensorB"].astype(int)
d2 = df['temperature_sensorA'].astype(int)
seconds = 5
fps = 30
rs = [crosscorr(d1,d2, lag) for lag in range(-int(seconds*fps),int(seconds*fps+1))]
offset = np.ceil(len(rs)/2)-np.argmax(rs)
f,ax=plt.subplots(figsize=(14,3))
ax.plot(rs)
ax.axvline(np.ceil(len(rs)/2),color='k',linestyle='--',label='Center')
ax.axvline(np.argmax(rs),color='r',linestyle='--',label='Peak synchrony')
ax.set(title=f'Offset = {offset} frames\nS1 leads <> S2 leads',ylim=[.1,.31],xlim=[0,301], xlabel='Offset',ylabel='Pearson r')
ax.set_xticks([0, 50, 100, 151, 201, 251, 301])
ax.set_xticklabels([-150, -100, -50, 0, 50, 100, 150]);
plt.legend()


# In[ ]:



