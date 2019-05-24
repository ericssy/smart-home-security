import tensorflow as tf
import numpy as np
import keras
import pandas as pd

class Autoencoder:
    def __init__(self, model, threshold):
        self.model = model
        self.threshold = threshold

    def predict(self, X):
        pred = self.model.predict(X)
        mse = np.square(np.subtract(pred, X))
        if mse < self.threshold:
            return 1
        else:
            return -1

def train():
    data = pd.read_csv("norm_data.csv")
    data = (data - data.min()) / (data.max()- data.min())
    X = data.values
    X_train = X[:int(len(X)*0.8)]
    X_test = X[int(len(X)*0.8):]
    n_features = X.shape[1]

    inp = keras.layers.Input(shape=(n_features,))
    enc = keras.layers.Dense(n_features // 2, activation='tanh')(inp) 
    dec = keras.layers.Dense(n_features, activation='tanh')(enc) 

    model = keras.models.Model(inputs=inp, outputs=dec)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc'])
    model.fit(X_train, X_train, epochs=50, batch_size=64)

    pred = model.predict(X_test)
    mse = []
    for i in len(pred)
        mse.append(np.sum(np.square(np.subtract(pred[i], X_test[i])))
    threshold = np.percentile(mse, 95)

    return Autoencoder(model, threshold)


