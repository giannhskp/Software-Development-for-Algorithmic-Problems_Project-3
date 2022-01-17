import sys
import os
import getopt
import pandas as pd
import matplotlib.pyplot as plt
import math
import keras
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from tensorflow.keras.optimizers import Adam


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

TRAIN_NEW_MODEL = False
PRINT_1_CURVE_MODELS_RESULTS = True

INPUT_SIZE = dataset.shape[0]
SERIES_LENGTH = dataset.shape[1]-1
TRAIN_LENGTH = math.floor(5*SERIES_LENGTH/10)
WINDOW_SIZE = 60
PREDICT_CURVES = list(range(number_of_tseries))
MAX_MODELS_FROM_1_CURVE = 20

training_set = dataset.iloc[:, 1:TRAIN_LENGTH+1].values
test_set = dataset.iloc[:, TRAIN_LENGTH+1:TRAIN_LENGTH*2+1].values

if TRAIN_NEW_MODEL:
    EPOCHS = 5
    BATCH_SIZE = 2048
    LEARNING_RATE = 0.01
    SHUFFLE_TRAIN_DATA = True
    # scale train data
    TRAIN_CURVES = list(range(INPUT_SIZE))
    training_set_reshaped = training_set[TRAIN_CURVES].reshape(-1, 1)

    sc = MinMaxScaler(feature_range=(0, 1))  # initialize scaler

    X_train = []
    y_train = []
    training_size = training_set_reshaped.shape[0]
    for i in range(WINDOW_SIZE, training_size):
        sc.fit(training_set_reshaped[i-WINDOW_SIZE:i+1, 0].reshape(-1, 1))
        x_transformed = sc.transform(training_set_reshaped[i-WINDOW_SIZE:i, 0].reshape(-1, 1))
        y_transformed = sc.transform(training_set_reshaped[i, 0].reshape(-1, 1))
        X_train.append(x_transformed)
        y_train.append(y_transformed)
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = Sequential()
    # Adding the first LSTM layer and some Dropout regularisation
    model.add(LSTM(units=128, return_sequences=True, input_shape=(WINDOW_SIZE, 1)))
    model.add(Dropout(0.2))
    # Adding a second LSTM layer and some Dropout regularisation
    model.add(LSTM(units=96, return_sequences=True))
    model.add(Dropout(0.2))
    # Adding a third LSTM layer and some Dropout regularisation
    model.add(LSTM(units=64, return_sequences=True))
    model.add(Dropout(0.2))
    # Adding a fourth LSTM layer and some Dropout regularisation
    model.add(LSTM(units=32))
    model.add(Dropout(0.2))
    # Adding the output layer
    model.add(Dense(units=1))

    # Compiling the RNN
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Fitting the RNN to the Training set
    history = model.fit(X_train, y_train,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        shuffle=SHUFFLE_TRAIN_DATA,
                        validation_split=0.1,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=3, mode='min')]
                        )
    model.save('part1_new_model.h5')  # creates a HDF5 file 'part1_new_model.h5'
else:
    # load saved trained model
    model = keras.models.load_model(model_loc)


sc = MinMaxScaler(feature_range=(0, 1))  # initialize scaler

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

    if curve < MAX_MODELS_FROM_1_CURVE and PRINT_1_CURVE_MODELS_RESULTS:
        temp_model_loc = r'models/part1/one_curve_models/part1_curve'+str(curve)+'.h5'
        model_from_one_curve = keras.models.load_model(temp_model_loc)
        predicted_stock_price_2 = model_from_one_curve.predict(X_test)
        unscaled_list = []
        for i, x in enumerate(predicted_stock_price_2):
            t_scale = scaler_list[i]
            inversed = np.array(t_scale.inverse_transform(x.reshape(-1, 1)))
            unscaled_list.append(inversed)
        unscaled = np.array(unscaled_list)
        unscaled = unscaled.reshape(-1, 1)
        predicted_stock_price_2 = unscaled

    result_size = X_test.shape[0]
    time = list(range(1, result_size+1))
    f = plt.figure()
    f.set_figwidth(25)
    f.set_figheight(4)
    plt.plot(time, test_set[curve], color='red', label='Real Curve')
    plt.plot(time, predicted_stock_price, color='blue', label='Predicted Curve')
    if curve < MAX_MODELS_FROM_1_CURVE and PRINT_1_CURVE_MODELS_RESULTS:
        plt.plot(time, predicted_stock_price_2, color='green', label='Predicted Curve from one-train-curve model')
    plt.xticks(np.arange(0, result_size, 50), rotation=70)
    plt.title('Prediction for curve: '+str(dataset.index[curve]))
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
plt.show()
