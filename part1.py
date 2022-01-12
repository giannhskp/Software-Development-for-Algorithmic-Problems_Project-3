import pandas as pd
import matplotlib.pyplot as plt
import math
import keras
import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
# from keras.layers import Dropout
# from keras.layers import *
# from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean_absolute_error
# from sklearn.model_selection import train_test_split
# from keras.callbacks import EarlyStopping

data_loc = r'../nasdaq2007_17.csv'
model_loc = r'part1_model'
dataset = pd.read_csv(data_loc, index_col=0, sep='\t', header=None)

INPUT_SIZE = dataset.shape[0]
SERIES_LENGTH = dataset.shape[1]-1
TRAIN_LENGTH = math.floor(5*SERIES_LENGTH/10)
EPOCHS = 5
BATCH_SIZE = 2048
LEARNING_RATE = 0.01
WINDOW_SIZE = 60
SHUFFLE_TRAIN_DATA = True
PREDICT_CURVES = list(range(INPUT_SIZE))
TRAIN_CURVES = list(range(INPUT_SIZE))

training_set = dataset.iloc[:, 1:TRAIN_LENGTH+1].values
test_set = dataset.iloc[:, TRAIN_LENGTH+1:TRAIN_LENGTH*2+1].values

sc = MinMaxScaler(feature_range=(0, 1))

model = keras.models.load_model(model_loc)

for curve in PREDICT_CURVES:
    dataset_train = dataset.iloc[curve:curve+1, 1:TRAIN_LENGTH+1]
    dataset_test = dataset.iloc[curve:curve+1, TRAIN_LENGTH+1:TRAIN_LENGTH*2+1]

    dataset_total = pd.concat((dataset_train, dataset_test), axis=1)
    inputs = dataset_total.iloc[:, dataset_total.shape[1] - dataset_test.shape[1] - WINDOW_SIZE:].values
    inputs = inputs.reshape(-1, 1)
    inputs = sc.fit_transform(inputs)
    X_test = []
    for i in range(WINDOW_SIZE, inputs.shape[0]):
        X_test.append(inputs[i-WINDOW_SIZE:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    result_size = X_test.shape[0]
    time = list(range(1, result_size+1))
    f = plt.figure()
    f.set_figwidth(15)
    f.set_figheight(4)
    plt.plot(time, test_set[curve], color='red', label='Real Curve')
    plt.plot(time, predicted_stock_price, color='blue', label='Predicted Curve')
    plt.xticks(np.arange(0, result_size, 50))
    plt.title('Prediction for curve: '+str(dataset.index[curve]))
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
