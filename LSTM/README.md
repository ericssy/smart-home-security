## Sensor fusion 

Malicious attacker device is able to send data that misleads the SmartHome sensors. Here in our experiment, we simulated temperature features and combine them with features from SmartHome sensor to confuse the detection. The attacker device has no access to the environment inside the room in which the sensors are located. Thus, the readings from the attacker device are not synchronized with the readings from the SmartHome sensors. We have multiple sensors that records the temperatures in the room. Due to differenes in the type, timing and location of the sensors, the readings are not exactly the same. However, most part of the temperature features from the SmartHome sensors are in some ways synchronized. Here, we try to detect the features coming from an attacker device by utilizing this property. 

The detection is made possible by measuring the cross validation on two time series data. We shifted one of the time series incrementally so that there is a lag between the two time series, and then measure the cross correlation. We picked the lag with the highest correlation. With that, we can know if the two time series are synchronized. A threshold on the offset is picked such that when the offset is greater than the threshold, the time series is considered as from a malicious device. 





## LSTM model

### 1.Feature Transformation

The `normal_data.csv` and `abnormal_data.csv` are transformed from the raw data collected by the sensors and  the simulated raw data respectively. `normal_data.csv` and `abnormal_data.csv` contains the status of temperature, door status, accleration and motion at each timestamp. The interval of each timestamp is not fixed because the sensor only updates the dataset once it detects a change.  

The two datasets, `normal_data.csv` and `abnormal_data.csv`  are transformed from `(sample_size * num_of_features)` to `(sample_size * num_of_steps * num_of_features)`. `num_of_steps` is the number of timestamps we trace back (from time $t_{i-20}$ to $t_i​$) in one sample. The number of steps in our experiment is 20 and the num of features is 5. Each sample in the transformed data is a 20 * 5 vector. 

The model is trained with the normal_data alone, with 80% of the samples as training data and 20% for validation. The trained model is trained to minimize the MSE with the adam optimizer. The validation data is used to determine a threshold on MSE. The threshold allows us to control the false positive rate. The final model predicted if an event is normal or abnormal based on the MSE threshold. If an event is abnormal, the event would most likely to deviate from the predicted output by the LSTM model, and thus would be considered as abnormal.

The abnormality detection rate is measured on a dataset with abormal data only. The overall accuracy is measured on a dataset with abnormal data inserted into normal behavior data to simulate real-life scenarios.



## Data with abnormal data inserted into normal data for overall accuracy

LSTM

testing with abnormal and normal data seperately:

- Overall accuracy: 91.0%
- abnormality accuracy: 83.1%
- false positive: 5.0%



LSTM autoencoder

- testing with abnormal and normal data seperately:
  - Overall Accuracy: 96.7% 
  - abnormality accuracy: 100%
  - false positive: 5.0%

* testing data with normal and abnormal behaviors
  * overall accuracy: 85.7%
  * abnormality accuracy: 47.7%
  * false positive rate: 14.3%





## Event Crawler

The normal event data is crawled with a script built with the Scrapy Framework. The html file of the event list is downloaded to the folder. The `base_url` variable in `event_2.py` is modified to the path of the downloaded file. After the configuration, go to the folder where you store the event_2.py file and run the command `scrapy crawl event -o [filename]` to crawl the event page and save the data as a csv file. 

Create a virtual env : python3 -m venv env
Activate your env : source env/bin/activate
Install Scrapy with pip : pip install scrapy
Start your crawler : ​sudo scrapy crawl event -o result.csv



##  Two newly simulated abnormal datasets

### 1. simulated_data.csv

##### Smaller time interval bewteen two detections 

In the original abnormal dataset, the average time difference between two detections (two consecutive samples) is 43.0 seconds, which is way much greater than the average 6.04 seconds in the normal dataset. I think the average time difference bewteen each detection in the abnormal dataset should be at similar or smaller than that in the normal dataset. The time inverval (in seconds) between two detection windows in the new simulated abnormal data is a beta distribution with $\alpha = 1.394, \beta = 9.54 * 10 ^{13}$, shifted and scaled with parameters $loc = -0.068, scale = 3.66 * 10^{14}$ (<https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html>). The average time difference in the new simulated data is 5.27 seconds

##### More frequent "Inactive" Status in the new abnormal data

Most of the Acceleration and Motion status in both the new and old abnormal data are 'inactive'. Having more "inactive" status could possibly make it harder to detect abnormality. Each detection can have up to two active status, one for the Acceleration and one for Motion. Detections (samples) with at least one 'inactive' are simulated to appear in a certain time frame, in a beta distribution with $\alpha = 0.571, \beta = 33.3, loc = 0.00285, scale = 1413.7$. All the detections (samples) in between have "active" status for both Acceleration and Motion. Detections with at least one 'inactive' are randomly selected from ["active", "inactive"], ["inactive", "active"], ["inactive", "inactive"] with probabilities 0.25, 0.25 and 0.5 respectively. 

#### Accuracy Comparison 

abnormality rate 

* original abnormal data
  * abnormality rate: 0.831 under 95% percentile threshold (false positive = 0.05)
  * abnormality rate: 0.909 under 93% percentile threshold (false positive = 0.07)
  * abnormality rate: 0.942 under 90% percentile threshold (false positive = 0.10)
* newly simulated abnormal data
  * abnormality rate: 0.836 under 95% percentile threshold (false positive = 0.05)
  * abnormality rate: 0.841 under 93% percentile threshold (false positive = 0.07)
  * abnormality rate: 0.942 under 90% percentile threshold (false positive = 0.10)



### 2. abnormal_with_simulated_normal_temp.csv

##### Simulated abnormal data with normal temperature, keep other columns unchanged:

There might be situations when there are abnormal activities and the temperature is normal. Here in this dataset, only the Temperature column is changed from the old abnormal data.

When simulated abnormal data with normal temperature is used as testing data, the abnormality accuracy (true positive rate) is 85.5%, under 95% percentile threshold. 

- ##### original abnormal data

  - abnormality rate: 0.831 under 95% percentile threshold (false positive = 0.05)
  - abnormality rate: 0.909 under 93% percentile threshold (false positive = 0.07)
  - abnormality rate: 0.942 under 90% percentile threshold (false positive = 0.10)

- ##### abnormal data with normal temperature

  - abnormality rate: 0.855 under 95% percentile threshold (false positive = 0.05)
  - abnormality rate: 0.864 under 93% percentile threshold (false positive = 0.07)
  - abnormality rate: 0.947 under 90% percentile threshold (false positive = 0.10)

  



