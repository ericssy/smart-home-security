##  Added two simulated abnormal datasets

### 1. simulated_data.csv

##### Smaller time interval bewteen two detections 

In the original abnormal dataset, the average time difference between two detections (two consecutive samples) is 43.0 seconds, which is way much greater than the average 6.04 seconds in the normal dataset. I think the average time difference bewteen each detection in the abnormal dataset should be at similar or smaller than that in the normal dataset. The time inverval (in seconds) between two detection windows in the new simulated abnormal data is a beta distribution with $\alpha = 1.394, \beta = 9.54 * 10 ^{13}​$, shifted and scaled with parameters $loc = -0.068, scale = 3.66 * 10^{14} ​$ (<https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html>). The average time difference in the new simulated data is 5.27 seconds

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

  



