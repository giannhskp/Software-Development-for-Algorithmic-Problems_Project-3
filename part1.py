import sys
import os
import getopt
import pandas as pd
import matplotlib.pyplot as plt
import math
import keras
# import tensorflow as tf
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
argv = sys.argv[1:]
data_loc = ''
number_of_tseries = -1
try:
    opts, args = getopt.getopt(argv, "d:n:")
except getopt.GetoptError:
    print('Wrong Arguments! \n Usage: $python forecast.py -d <dataset> -n <number of time series selected>')
    sys.exit(2)
for opt, arg in opts:
    if opt == "-d":
        data_loc = arg
    elif opt == "-n":
        number_of_tseries = int(arg)
if data_loc != '':
    print('Dataset file is ', data_loc)
else:
    print('Dataset file was not given! \n Usage: $python forecast.py -d <dataset> -n <number of time series selected>')
if number_of_tseries > 0:
    print('Number of time series selected is: ', number_of_tseries)
else:
    number_of_tseries = 5
    print('Number of time series selected was not given, default value is: ', number_of_tseries)

# data_loc = r'input_files/nasdaq2007_17.csv'
model_loc = r'models/part1/part1_all_curves.h5'
dataset = pd.read_csv(data_loc, index_col=0, sep='\t', header=None)

INPUT_SIZE = dataset.shape[0]
SERIES_LENGTH = dataset.shape[1]-1
TRAIN_LENGTH = math.floor(5*SERIES_LENGTH/10)
EPOCHS = 5
BATCH_SIZE = 2048
LEARNING_RATE = 0.01
WINDOW_SIZE = 60
SHUFFLE_TRAIN_DATA = True
PREDICT_CURVES = list(range(number_of_tseries))
TRAIN_CURVES = list(range(INPUT_SIZE))
MAX_MODELS_FROM_1_CURVE = 20

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
    # inputs = sc.fit_transform(inputs)
    X_test = []
    scaler_list = []
    for i in range(WINDOW_SIZE, inputs.shape[0]):
        temp_scaler = MinMaxScaler(feature_range=(0, 1))
        transformed = temp_scaler.fit_transform(inputs[i-WINDOW_SIZE:i, 0].reshape(-1, 1))
        scaler_list.append(temp_scaler)
        X_test.append(transformed)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predicted_stock_price = model.predict(X_test)
    unscaled_list = []
    for i, x in enumerate(predicted_stock_price):
        t_scale = scaler_list[i]
        inversed = np.array(t_scale.inverse_transform(x.reshape(-1, 1)))
        unscaled_list.append(inversed)
    unscaled = np.array(unscaled_list)
    unscaled = unscaled.reshape(-1, 1)
    predicted_stock_price = unscaled

    # if curve < MAX_MODELS_FROM_1_CURVE:
    #     temp_model_loc = r'models/part1_curve'+str(curve)+'.h5'
    #     model_from_one_curve = keras.models.load_model(temp_model_loc)
    #     predicted_stock_price_2 = model_from_one_curve.predict(X_test)
    #     unscaled_list = []
    #     for i, x in enumerate(predicted_stock_price_2):
    #         t_scale = scaler_list[i]
    #         inversed = np.array(t_scale.inverse_transform(x.reshape(-1, 1)))
    #         unscaled_list.append(inversed)
    #     unscaled = np.array(unscaled_list)
    #     unscaled = unscaled.reshape(-1, 1)
    #     predicted_stock_price_2 = unscaled

    result_size = X_test.shape[0]
    time = list(range(1, result_size+1))
    f = plt.figure()
    f.set_figwidth(15)
    f.set_figheight(4)
    plt.plot(time, test_set[curve], color='red', label='Real Curve')
    plt.plot(time, predicted_stock_price, color='blue', label='Predicted Curve')
    # if curve < MAX_MODELS_FROM_1_CURVE:
    #     plt.plot(time, predicted_stock_price_2, color='green', label='Predicted Curve from one-train-curve model')
    plt.xticks(np.arange(0, result_size, 50), rotation=70)
    plt.title('Prediction for curve: '+str(dataset.index[curve]))
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
plt.show()
