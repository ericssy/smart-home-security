import pickle
import numpy as np
from sklearn import svm

class OneClassSVM_Ab:
    def __init__(self, file_name):
        self.clf = pickle.load(open(file_name, 'rb'), encoding='latin1')

    # temp_change_f: times of temperature changes within the 100s window
    # temp_avg_f: average temperature within the 100s window
    # door_cnt_f: count of active door status within the 100s window
    # motion_cnt_f: count of active motion status within the 100s window
    # acc_cnt_f: count of active acceleration status within the 100s window
    # timestamp_f: ignore date and the way to get timestamp is: 
    #   (1) convert time of the middle time point of the 100s window to the i-th "second" within a day. For example, if the time is 16:23:15, the "second" is (16 * 60 + 23) * 60 + 15 = 58995
    #   (2) map it to the 48 time zones in a day, which means each zone include 1200s. Therefore, the way to map is: 58995 / 1200
    def predict(self, temp_change_f, temp_avg_f, door_cnt_f, motion_cnt_f, acc_cnt_f, timestamp_f):
        X_list = []
        X_first = []
        
        X_first.append(temp_change_f)
        X_first.append(temp_avg_f)
        X_first.append(door_cnt_f)
        X_first.append(motion_cnt_f)
        X_first.append(acc_cnt_f)
        X_first.append(timestamp_f)
        X_list.append(X_first)
        X = np.array(X_list)
        
        return self.clf.predict(X)
        
