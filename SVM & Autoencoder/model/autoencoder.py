import tensorflow as tf
import numpy as np
import keras
import pandas as pd

def clock_to_seconds(clock_time):
    lst = clock_time.split(':')
    secs = 0
    t = 1
    for i in range(2, -1, -1):
        secs += float(lst[i]) * t
        t *= 60
    return secs



class Autoencoder:
    def __init__(self, model, threshold):
        self.model = model
        self.threshold = threshold

    def predict(self, X):
        pred = self.model.predict(np.array([X,]))
        mse = np.sum(np.square(np.subtract(pred, X)))
        if mse < self.threshold:
            return 1
        else:
            return 0

def process_data(file_name):
    data = pd.read_csv(file_name)
    data['Motion'] = data['Motion'].map({'inactive': 0, 'active': 1})
    data['Acceleration'] = data['Acceleration'].map({'inactive': 0, 'active': 1})
    data['Time'] = data['Time'].apply(clock_to_seconds) / 86400.0
    data['Temperature'] = (data['Temperature'] - 65.0) / 20.0
    data['DayOfWeek'] = data['DayOfWeek'] / 6.0
    data = data.clip(0, 1)
    return data

def train():
    data = process_data('normal_data.csv')
    X = data.values
    X_train = X[:int(len(X)*0.8)]
    X_test = X[int(len(X)*0.8):]
    n_features = X.shape[1]

    inp = keras.layers.Input(shape=(n_features,))
    enc = keras.layers.Dense(n_features-2, activation='tanh')(inp) 
    dec = keras.layers.Dense(n_features, activation='tanh')(enc) 

    model = keras.models.Model(inputs=inp, outputs=dec)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc'])
    model.fit(X_train, X_train, epochs=20, batch_size=50)

    pred = model.predict(X_test)
    mse = []
    for i in range(len(pred)):
        mse.append(np.sum(np.square(np.subtract(pred[i], X_test[i]))))
    threshold = np.percentile(mse, 95)
    print(threshold)

    return Autoencoder(model, threshold)

autoenc = train()
data = process_data('abnormal_data.csv')
X = data.values
X_test = X[:int(len(X)*0.5)]

t = 0
for i in range(len(X_test)):
    t += autoenc.predict(X_test[i])
print(t)

