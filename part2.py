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
from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean_absolute_error
# from sklearn.model_selection import train_test_split
# from keras.callbacks import EarlyStopping

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
argv = sys.argv[1:]
data_loc = ''
number_of_tseries = -1
THRESHOLD = -1
THRESHOLD_BEST_VALUE = 2.25
try:
    opts, args = getopt.getopt(argv, "d:n:m:")
except getopt.GetoptError:
    print('Wrong Arguments! \n Usage: $python forecast.py -d <dataset> -n <number of time series selected>')
    sys.exit(2)
for opt, arg in opts:
    if opt == "-d":
        data_loc = arg
    elif opt == "-n":
        number_of_tseries = int(arg)
    elif opt == "-mae":
        THRESHOLD = int(arg)
if data_loc != '':
    print('Dataset file is ', data_loc)
else:
    print('Dataset file was not given! \n Usage: $python forecast.py -d <dataset> -n <number of time series selected>')
if number_of_tseries > 0:
    print('Number of time series selected is: ', number_of_tseries)
else:
    number_of_tseries = 5
    print('Number of time series selected was not given, default value is: ', number_of_tseries)
if THRESHOLD > 0:
    print('MAE (thresdhold) is set to : ', THRESHOLD, ' (optimal value = )', THRESHOLD_BEST_VALUE)
else:
    THRESHOLD = THRESHOLD_BEST_VALUE
    print('MAE (thresdhold) was not given, default/optimal value is: ', THRESHOLD)

dataset = pd.read_csv(data_loc, index_col=0, sep='\t', header=None)

model_loc = 'models/part2/part2_200curves.h5'
INPUT_SIZE = dataset.shape[0]
SERIES_LENGTH = dataset.shape[1]-1
TRAIN_LENGTH = math.floor(5*SERIES_LENGTH/10)
EPOCHS = 20
BATCH_SIZE = 2048
LEARNING_RATE = 1e-3
WINDOW_SIZE = 60
SHUFFLE_TRAIN_DATA = True
samples = list(range(number_of_tseries))
PREDICT_CURVES = samples
TRAIN_CURVES = samples
PLOT_MAE = False

training_set = dataset.iloc[:, 1:TRAIN_LENGTH+1].values
test_set = dataset.iloc[:, TRAIN_LENGTH+1:TRAIN_LENGTH*2+1].values

sc = StandardScaler()

model = keras.models.load_model(model_loc)

if PLOT_MAE:
    training_set_scaled = training_set[TRAIN_CURVES].reshape(-1, 1)
    X_train = []
    y_train = []
    for j in range(0, len(TRAIN_CURVES)):
        down_range = j*TRAIN_LENGTH+WINDOW_SIZE
        up_range = (j+1)*TRAIN_LENGTH
        for i in range(down_range, up_range):
            sc.fit(training_set_scaled[i-WINDOW_SIZE:i+1, 0].reshape(-1, 1))
            x_transformed = sc.transform(training_set_scaled[i-WINDOW_SIZE:i, 0].reshape(-1, 1))
            y_transformed = sc.transform(training_set_scaled[i, 0].reshape(-1, 1))
            X_train.append(x_transformed)
            y_train.append(y_transformed)
    X_train, y_train = np.array(X_train), np.array(y_train)
    # predict on train set to find the threshold
    X_train_pred = model.predict(X_train)
    train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)
    plt.hist(train_mae_loss, bins=50)
    plt.xlabel('Train MAE loss')
    plt.ylabel('Number of Samples')

    threshold_max = np.max(train_mae_loss)
    threshold_mean = np.mean(train_mae_loss)
    print(f'MAX error threshold: {threshold_max}')
    print(f'MEAN error threshold: {threshold_mean}')

for curve in PREDICT_CURVES:
    dataset_train = dataset.iloc[curve:curve+1, 1:TRAIN_LENGTH+1]
    dataset_test = dataset.iloc[curve:curve+1, TRAIN_LENGTH+1:TRAIN_LENGTH*2+1]

    dataset_total = pd.concat((dataset_train, dataset_test), axis=1)
    inputs = dataset_total.iloc[:, dataset_total.shape[1] - dataset_test.shape[1] - WINDOW_SIZE:].values
    inputs = inputs.reshape(-1, 1)
    # inputs = sc.transform(inputs)
    X_test = []
    for i in range(WINDOW_SIZE, inputs.shape[0]):
        x_transformed = sc.fit_transform(inputs[i-WINDOW_SIZE:i, 0].reshape(-1, 1))
        X_test.append(x_transformed)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    X_test_pred = model.predict(X_test)
    test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)

    test = dataset_total.iloc[:, dataset_total.shape[1] - dataset_test.shape[1] - WINDOW_SIZE:].T
    test_score_df = pd.DataFrame(index=test[WINDOW_SIZE:].index)
    test_score_df['loss'] = test_mae_loss
    test_score_df['threshold'] = THRESHOLD
    test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
    test_score_df['close'] = test[WINDOW_SIZE:]

    result_size = X_test.shape[0]
    time = list(range(1, result_size+1))
    # f = plt.figure()
    # f.set_figwidth(20)
    # f.set_figheight(5)
    # plt.plot(time, test_score_df.threshold, color='red', label='Threshold')
    # plt.plot(time, test_score_df.loss, color='blue', label='Loss')
    # plt.xticks(np.arange(0, result_size, 25), rotation=70)
    # plt.title('Loss vs Threshold for curve: '+str(dataset.index[curve]))
    # plt.xlabel('Time')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()

    anomalies = test_score_df[test_score_df.anomaly == True]
    anomalies_indexes = anomalies.index-test_score_df.index[0]
    anomalies_indexes = anomalies_indexes.tolist()
    anomalies_times = [time[i] for i in anomalies_indexes]
    f = plt.figure()
    f.set_figwidth(20)
    f.set_figheight(5)
    plt.plot(time, test_score_df.close, color='red', label='Real Curve')
    plt.scatter(anomalies_times, anomalies.close, color='blue', label='Anomalies')
    plt.xticks(np.arange(0, result_size, 50), rotation=70)
    plt.title('Anomalies of curve: '+str(dataset.index[curve]))
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
plt.show()
