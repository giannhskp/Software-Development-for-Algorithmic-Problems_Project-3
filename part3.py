import sys
import os
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
argv = sys.argv[1:]
data_loc = ''
query_loc = ''
out_data_loc = ''
out_query_loc = ''
number_of_tseries = -1
THRESHOLD = -1
THRESHOLD_BEST_VALUE = 2.25

for i, arg in enumerate(argv):
    if arg == "-d":
        data_loc = argv[i+1]
    elif arg == "-q":
        query_loc = argv[i+1]
    elif arg == "-od":
        out_data_loc = argv[i+1]
    elif arg == "-oq":
        out_query_loc = argv[i+1]
if data_loc == '' or query_loc == '' or out_data_loc == '' or out_query_loc == '':
    print('Wrong Arguments! \n Usage: $python reduce.py –d  <dataset> -q <queryset> -od <output_dataset_file> -oq <output_query_file>')
    sys.exit(2)
else:
    print('Dataset file is ', data_loc)
    print('Query file is ', query_loc)
    print('Dataset output file is ', out_data_loc)
    print('Query output is ', out_query_loc)


dataset = pd.read_csv(data_loc, index_col=0, sep='\t', header=None)
queryset = pd.read_csv(query_loc, index_col=0, sep='\t', header=None)
print(dataset.shape)
print(queryset.shape)

encoder_loc = 'models/part3/part3_encoder.h5'
autoencoder_loc = 'models/part3/part3_autoencoder.h5'

encoder = keras.models.load_model(encoder_loc)
autoencoder = keras.models.load_model(autoencoder_loc)

WINDOW_SIZE = 10
query_result_list = []
for curve in range(queryset.shape[0]):
    original_curve = queryset.iloc[curve:curve+1, :].values
    original_curve = original_curve.reshape(-1, 1)
    df = pd.DataFrame(np.array(original_curve)[:, 0], columns=['price'])
    loop_range = range(0, math.floor((len(df['price'])-1)/WINDOW_SIZE))
    scaler_list = []
    transformed_list = []
    for i in loop_range:
        temp_scaler = MinMaxScaler()
        transformed = np.array(temp_scaler.fit_transform(df['price'].values[i*WINDOW_SIZE+1:(i+1)*WINDOW_SIZE+1].reshape(-1, 1)))

        scaler_list.append(temp_scaler)
        transformed_list.append(transformed)
    x_transform = np.array(transformed_list)

    tranformed = encoder.predict(x_transform)

    unscaled_list = []
    for i, x in enumerate(tranformed):
        t_scale = scaler_list[i]
        inversed = np.array(t_scale.inverse_transform(x))
        unscaled_list.append(inversed)
    unscaled = np.array(unscaled_list)
    unscaled = unscaled.reshape(-1, 1)
    query_result_list.append(unscaled)

query_result = np.array(query_result_list)
query_result = query_result.reshape(query_result.shape[0], query_result.shape[1])
query_df = pd.DataFrame(data=query_result, index=queryset.index[:])
query_df.to_csv(out_query_loc, sep='\t', encoding='utf-8', header=False)


dataset_result_list = []
for curve in range(dataset.shape[0]):
    original_curve = dataset.iloc[curve:curve+1, :].values
    original_curve = original_curve.reshape(-1, 1)
    df = pd.DataFrame(np.array(original_curve)[:, 0], columns=['price'])
    loop_range = range(0, math.floor((len(df['price'])-1)/WINDOW_SIZE))
    scaler_list = []
    transformed_list = []
    for i in loop_range:
        temp_scaler = MinMaxScaler()
        transformed = np.array(temp_scaler.fit_transform(df['price'].values[i*WINDOW_SIZE+1:(i+1)*WINDOW_SIZE+1].reshape(-1, 1)))

        scaler_list.append(temp_scaler)
        transformed_list.append(transformed)
    x_transform = np.array(transformed_list)

    tranformed = encoder.predict(x_transform)

    unscaled_list = []
    for i, x in enumerate(tranformed):
        t_scale = scaler_list[i]
        inversed = np.array(t_scale.inverse_transform(x))
        unscaled_list.append(inversed)
    unscaled = np.array(unscaled_list)
    unscaled = unscaled.reshape(-1, 1)
    dataset_result_list.append(unscaled)

dataset_result = np.array(dataset_result_list)
dataset_result = dataset_result.reshape(dataset_result.shape[0], dataset_result.shape[1])
dataset_df = pd.DataFrame(data=dataset_result, index=dataset.index[:])
dataset_df.to_csv(out_data_loc, sep='\t', encoding='utf-8', header=False)
