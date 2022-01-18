import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import math
import keras
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
argv = sys.argv[1:]
data_loc = ''
number_of_tseries = -1
THRESHOLD = -1
THRESHOLD_BEST_VALUE = 2.25
PLOT_MAE = False

for i, arg in enumerate(argv):
    if arg == "-d":
        data_loc = argv[i+1]
    elif arg == "-n":
        number_of_tseries = int(argv[i+1])
    elif arg == "-mae":
        THRESHOLD = int(argv[i+1])
    elif arg == "-print_mae":
        PLOT_MAE = True
if data_loc != '':
    print('Dataset file is ', data_loc)
else:
    print('Dataset file was not given! \n Usage: $python detect.py -d <dataset> -n <number of time series selected> -mae <error value as double>')
if number_of_tseries > 0:
    print('Number of time series selected is: ', number_of_tseries)
else:
    number_of_tseries = 5
    print('Number of time series selected was not given, default value is: ', number_of_tseries)
if THRESHOLD > 0:
    print('MAE (thresdhold) is set to : ', THRESHOLD, ' (optimal value = ', THRESHOLD_BEST_VALUE, ')')
else:
    THRESHOLD = THRESHOLD_BEST_VALUE
    print('MAE (thresdhold) was not given, default/optimal value is: ', THRESHOLD)

dataset = pd.read_csv(data_loc, index_col=0, sep='\t', header=None)

TRAIN_NEW_MODEL = False

model_loc = 'models/part2/part2_200curves.h5'
INPUT_SIZE = dataset.shape[0]
SERIES_LENGTH = dataset.shape[1]-1
TRAIN_LENGTH = math.floor(5*SERIES_LENGTH/10)

WINDOW_SIZE = 60
samples = list(range(number_of_tseries))
PREDICT_CURVES = samples
TRAIN_CURVES = samples

training_set = dataset.iloc[:, 1:TRAIN_LENGTH+1].values
test_set = dataset.iloc[:, TRAIN_LENGTH+1:TRAIN_LENGTH*2+1].values

if TRAIN_NEW_MODEL:
    print('Training new model...')
    EPOCHS = 5
    BATCH_SIZE = 2048
    LEARNING_RATE = 1e-3
    SHUFFLE_TRAIN_DATA = True
    training_set_reshaped = training_set[TRAIN_CURVES].reshape(-1, 1)

    sc = StandardScaler()
    X_train = []
    y_train = []
    for j in range(0, len(TRAIN_CURVES)):
        down_range = j*TRAIN_LENGTH+WINDOW_SIZE
        up_range = (j+1)*TRAIN_LENGTH
        for i in range(down_range, up_range):
            sc.fit(training_set_reshaped[i-WINDOW_SIZE:i+1, 0].reshape(-1, 1))
            x_transformed = sc.transform(training_set_reshaped[i-WINDOW_SIZE:i, 0].reshape(-1, 1))
            y_transformed = sc.transform(training_set_reshaped[i, 0].reshape(-1, 1))
            X_train.append(x_transformed)
            y_train.append(y_transformed)
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = keras.Sequential()
    model.add(keras.layers.LSTM(
        units=128,
        input_shape=(X_train.shape[1], X_train.shape[2]),
        return_sequences=True,
        activation='relu'
    ))
    model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.LSTM(units=64, return_sequences=False, activation='relu'))
    model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.RepeatVector(n=X_train.shape[1]))
    model.add(keras.layers.LSTM(units=64, return_sequences=True, activation='relu'))
    model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.LSTM(units=128, return_sequences=True, activation='relu'))
    model.add(keras.layers.Dropout(rate=0.2))
    model.add(
      keras.layers.TimeDistributed(
        keras.layers.Dense(units=X_train.shape[2])
      )
    )
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(loss='mae', optimizer=optimizer)
    model.summary()

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        shuffle=SHUFFLE_TRAIN_DATA,
        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')]
    )
    model.save('part2_new_model.h5')  # creates a HDF5 file 'part1_new_model.h5'
    print('Finished training. New model saved to: part2_new_model.h5')
else:
    model = keras.models.load_model(model_loc)

sc = StandardScaler()

if PLOT_MAE:
    if not TRAIN_NEW_MODEL:
        training_set_reshaped = training_set[TRAIN_CURVES].reshape(-1, 1)
        X_train = []
        y_train = []
        for j in range(0, len(TRAIN_CURVES)):
            down_range = j*TRAIN_LENGTH+WINDOW_SIZE
            up_range = (j+1)*TRAIN_LENGTH
            for i in range(down_range, up_range):
                sc.fit(training_set_reshaped[i-WINDOW_SIZE:i+1, 0].reshape(-1, 1))
                x_transformed = sc.transform(training_set_reshaped[i-WINDOW_SIZE:i, 0].reshape(-1, 1))
                y_transformed = sc.transform(training_set_reshaped[i, 0].reshape(-1, 1))
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
