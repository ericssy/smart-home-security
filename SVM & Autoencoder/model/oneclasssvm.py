import csv
import pickle
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score

def read_data():
    temp = 68
    door = 0
    acc = 0
    motion = 0

    temp_list = []
    door_list = []
    acc_list = []
    motion_list = []
    time_list = []
    # day_list = []

    types = []
    values = []
    times = []

    with open('data.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            types.append(row[0])

            value = row[1]
            if value == 'closed' or value == 'inactive':
                value = 0
            elif value == 'open' or value == 'active':
                value = 1
            else:
                value = int(value)
            values.append(value)

            time = row[2].split()[1].split(':')
            h = (int(time[0]) - 5) % 24
            m = int(time[1])
            s = int(time[2].split('.')[0])
            times.append((h * 60 + m) * 60 + s)

    rec_idx = len(types) - 1

    #for time in range(0, 86400):
    time = times[-1]
    #day = 3
    while True:
        while time == times[rec_idx]:
            if types[rec_idx] == 'temperature':
                temp = values[rec_idx]
            elif types[rec_idx] == 'status':
                door = values[rec_idx]
            elif types[rec_idx] == 'acceleration':
                acc = values[rec_idx]
            elif types[rec_idx] == 'motion':
                motion = values[rec_idx]

            if rec_idx > 0:
                rec_idx = rec_idx - 1
            else:
                break

        time_list.append(time)
        temp_list.append(temp)
        door_list.append(door)
        acc_list.append(acc)
        motion_list.append(motion)
        #day_list.append(day)

        if rec_idx == 0:
            break;
        time = time + 1
        if time == 86400:
            time = 0
            #day = day % 7 + 1

    return temp_list, door_list, acc_list, motion_list, time_list #, day_list

def get_feature():
    temp_change_f = []
    temp_avg_f = []
    door_cnt_f = []
    acc_cnt_f = []
    motion_cnt_f = []
    timestamp_f = []
    # day_f = []

    window_size = 100
    zone_size = 1200

    temp_change = 0
    temp_avg = 0
    door_cnt = 0
    acc_cnt = 0
    motion_cnt = 0

    # temp_list, door_list, acc_list, motion_list, time_list, day_list = read_data()
    temp_list, door_list, acc_list, motion_list, time_list = read_data()

    last_temp = 68

    for i in range(0, window_size):
        temp_avg = temp_avg + temp_list[i]
        if last_temp != temp_list[i]:
            temp_change = temp_change + 1
            last_temp = temp_list[i]
        if door_list[i] == 1:
            door_cnt = door_cnt + 1
        if acc_list[i] == 1:
            acc_cnt = acc_cnt + 1
        if motion_list[i] == 1:
            motion_cnt = motion_cnt + 1

    temp_change_f.append(temp_change)
    temp_avg_f.append(float(temp_avg) / window_size)
    door_cnt_f.append(door_cnt)
    acc_cnt_f.append(acc_cnt)
    motion_cnt_f.append(motion_cnt)
    timestamp_f.append((time_list[window_size / 2 - 1] % 86400) / zone_size)
    # day_f.append(day_list[window_size / 2 - 1])

    for i in range(window_size, len(time_list)):
        temp_avg = temp_avg + temp_list[i]
        if last_temp != temp_list[i]:
            temp_change = temp_change + 1
            last_temp = temp_list[i]
        if door_list[i] == 1:
            door_cnt = door_cnt + 1
        if acc_list[i] == 1:
            acc_cnt = acc_cnt + 1
        if motion_list[i] == 1:
            motion_cnt = motion_cnt + 1

        temp_avg = temp_avg - temp_list[i - window_size + 1]
        if temp_list[i - window_size] != temp_list[i - window_size + 1]:
            temp_change = temp_change - 1
        if door_list[i - window_size + 1] == 1:
            door_cnt = door_cnt - 1
        if acc_list[i - window_size + 1] == 1:
            acc_cnt = acc_cnt - 1
        if motion_list[i - window_size + 1] == 1:
            motion_cnt = motion_cnt - 1

        temp_change_f.append(temp_change)
        temp_avg_f.append(float(temp_avg) / window_size)
        door_cnt_f.append(door_cnt)
        acc_cnt_f.append(acc_cnt)
        motion_cnt_f.append(motion_cnt)
        timestamp_f.append(((time_list[i] - window_size / 2) % 86400) / zone_size)
        # day_f.append(day_list[i])

    X_list = []
    X_list.append(temp_change_f)
    X_list.append(temp_avg_f)
    X_list.append(door_cnt_f)
    X_list.append(motion_cnt_f)
    X_list.append(acc_cnt_f)
    X_list.append(timestamp_f)
    # X_list.append(day_f)
    X = np.array(X_list)

    X = np.swapaxes(X, 0, 1)
    return X

if __name__ == '__main__':
    X_train = get_feature()

    #print X_train.shape
    print('Start training.')
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf.fit(X_train)

    print('Start predicting.')
    y_pred_train = clf.predict(X_train)
    y_true = [1] * X_train.shape[0]

    print('accuracy: ' + str(accuracy_score(y_true, y_pred_train)))

    print('Save model.')
    pickle.dump(clf, open('model.p', 'wb'))