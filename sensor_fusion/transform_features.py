
# coding: utf-8

# In[41]:

import pandas as pd
import numpy as np
import time
import datetime


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

# In[42]:


sensorA = pd.read_csv("data_multipurpose_sensor.csv")
sensorB = pd.read_csv("data_motion_sensor.csv")
sensorC = pd.read_csv("data_multisensor.csv")


# In[43]:

sensorA 


# In[44]:

sensorB


# In[45]:

sensorC


# In[46]:

sensorA["name"] = sensorA["name"] + "_sensorA"


# In[47]:

sensorA["name"]


# In[48]:

sensorA


# In[ ]:




# In[49]:

sensorB["name"] = sensorB["name"] + "_sensorB"
sensorC["name"] = sensorC["name"] + "_sensorC"


# combine the three datasets 

# In[50]:

data = pd.concat([sensorA, sensorB, sensorC])


# In[ ]:




# rank the data by time, with ascending order 
# first conver the time to timestamp

# In[52]:




# In[55]:


timestamp_list = []
for i in range(data.shape[0]):
    #epoch = int(time.mktime(datetime.datetime.strptime(data["time"].iloc[i],"%Y-%m-%d %I:%M:%S.%f %p")))
    timestamp = datetime.datetime.strptime(data["time"].iloc[i],"%Y-%m-%d %I:%M:%S.%f %p").timestamp()
    timestamp_list.append(timestamp)
timestamp_list


# In[58]:

data["timestamp"] = timestamp_list


# In[60]:

data = data.sort_values(by = ["timestamp"], ascending = True)
data = data.reset_index(drop = True)


# In[ ]:

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

    
    


# In[ ]:




# In[ ]:



