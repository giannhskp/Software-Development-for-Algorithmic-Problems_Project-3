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
argv = sys.argv[1:]  # get parameters from command line
data_loc = ''  # location of dataset file given by the user
number_of_tseries = -1  # number of time series from the given dataset file that the autoencoder will be applied
THRESHOLD = -1
THRESHOLD_BEST_VALUE = 2.25
PLOT_MAE = False  # default, don't print the mae histogram

for i, arg in enumerate(argv):  # read parameters
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

TRAIN_NEW_MODEL = False  # don't train a new model, load the saved one

model_loc = 'models/part2/part2_model.h5'
INPUT_SIZE = dataset.shape[0]  # the number of time series of the dataset
SERIES_LENGTH = dataset.shape[1]-1  # the length of each time series
TRAIN_LENGTH = math.floor(5*SERIES_LENGTH/10)  # the length of each time series that will be used in train set (50%)

WINDOW_SIZE = 60
samples = list(range(number_of_tseries))
PREDICT_CURVES = samples
TRAIN_CURVES = samples

# create train set, using the first 50% of each curve
training_set = dataset.iloc[:, 1:TRAIN_LENGTH+1].values
# create test set, using the rest 50% of each curve
test_set = dataset.iloc[:, TRAIN_LENGTH+1:TRAIN_LENGTH*2+1].values

if TRAIN_NEW_MODEL:
    print('Training new model...')
    # hyper parameters
    EPOCHS = 5
    BATCH_SIZE = 2048
    LEARNING_RATE = 1e-3
    SHUFFLE_TRAIN_DATA = True

    training_set_reshaped = training_set[TRAIN_CURVES].reshape(-1, 1)

    # Let's create the corresponding lists (X_train and y_train) that will be used for the training of the model
    # In order to create the X_train, the time series are divided into successive sets of length equal to WINDOW_SIZE.
    # (ex. first set: [0,WINDOW_SIZE], second set: [1,WINDOW_SIZE+1], ...)
    # Each set of values of X_train corresponds to a value in y_train.
    # This value is the next value of the time series.
    sc = StandardScaler()  # initialize scaler
    X_train = []
    y_train = []
    for j in range(0, len(TRAIN_CURVES)):  # for every curve of the train set
        down_range = j*TRAIN_LENGTH+WINDOW_SIZE
        up_range = (j+1)*TRAIN_LENGTH
        for i in range(down_range, up_range):  # for every window of the curve
            sc.fit(training_set_reshaped[i-WINDOW_SIZE:i+1, 0].reshape(-1, 1))  # fit scaler to this window
            x_transformed = sc.transform(training_set_reshaped[i-WINDOW_SIZE:i, 0].reshape(-1, 1))  # apply scaling
            y_transformed = sc.transform(training_set_reshaped[i, 0].reshape(-1, 1))  # apply scaling
            X_train.append(x_transformed)  # add the scaled window to the list
            y_train.append(y_transformed)  # add the scaled next value of the window to the list
    X_train, y_train = np.array(X_train), np.array(y_train)  # convert the lists into NumPy arrays
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # MODEL DEFINITION #
    model = keras.Sequential()
    # Encoder
    model.add(keras.layers.LSTM(
        units=128,
        input_shape=(X_train.shape[1], X_train.shape[2]),
        return_sequences=True,
        activation='relu'
    ))
    model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.LSTM(units=64, return_sequences=False, activation='relu'))
    model.add(keras.layers.Dropout(rate=0.2))
    ##########################
    model.add(keras.layers.RepeatVector(n=X_train.shape[1]))  # creates a tensor of size (WINDOW_SIZE,64) which is given as input to the first LSTM layer of the decoder
    ##########################
    # Decoder
    model.add(keras.layers.LSTM(units=64, return_sequences=True, activation='relu'))
    model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.LSTM(units=128, return_sequences=True, activation='relu'))
    model.add(keras.layers.Dropout(rate=0.2))
    ##########################
    # finally, restore the time series at their initial dimensions
    model.add(
      keras.layers.TimeDistributed(
        keras.layers.Dense(units=X_train.shape[2])
      )
    )
    # Compiling the Autoencoder
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(loss='mae', optimizer=optimizer)
    model.summary()

    # Fitting the Autoencoder to the Training set
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
    # load the saved pre-trained model
    model = keras.models.load_model(model_loc)

sc = StandardScaler()  # initialize the scaler

if PLOT_MAE:
    # then print the mae histogram
    if not TRAIN_NEW_MODEL:
        training_set_reshaped = training_set[TRAIN_CURVES].reshape(-1, 1)
        # doing same procedure as above
        X_train = []
        y_train = []
        for j in range(0, len(TRAIN_CURVES)):
            down_range = j*TRAIN_LENGTH+WINDOW_SIZE
            up_range = (j+1)*TRAIN_LENGTH
            for i in range(down_range, up_range):
                sc.fit(training_set_reshaped[i-WINDOW_SIZE:i+1, 0].reshape(-1, 1))  # fit scaler
                x_transformed = sc.transform(training_set_reshaped[i-WINDOW_SIZE:i, 0].reshape(-1, 1))  # apply scaling
                y_transformed = sc.transform(training_set_reshaped[i, 0].reshape(-1, 1))  # apply scaling
                X_train.append(x_transformed)  # add the scaled window to the list
                y_train.append(y_transformed)  # add the scaled window to the list
        X_train, y_train = np.array(X_train), np.array(y_train)  # convert the lists into NumPy arrays

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
    # get the first 50% of the current curve
    dataset_train = dataset.iloc[curve:curve+1, 1:TRAIN_LENGTH+1]
    # get the rest 50% of the current curve
    dataset_test = dataset.iloc[curve:curve+1, TRAIN_LENGTH+1:TRAIN_LENGTH*2+1]
    # concatenate dataset_train and dataset_test
    dataset_total = pd.concat((dataset_train, dataset_test), axis=1)
    # finally keep the first half - WINDOW_SIZE of the curve
    inputs = dataset_total.iloc[:, dataset_total.shape[1] - dataset_test.shape[1] - WINDOW_SIZE:].values
    inputs = inputs.reshape(-1, 1)

    # Let's create the X_test list that will be give as input into the model.
    # In order to create the X_train, the current curve is divided into successive sets of length equal to WINDOW_SIZE.
    # (ex. first set: [0,WINDOW_SIZE], second set: [1,WINDOW_SIZE+1], ...)
    X_test = []
    for i in range(WINDOW_SIZE, inputs.shape[0]):
        x_transformed = sc.fit_transform(inputs[i-WINDOW_SIZE:i, 0].reshape(-1, 1))  # apply scaling
        X_test.append(x_transformed)  # add the scaled window to the list
    # finally X_test contains all the scaled windows
    X_test = np.array(X_test)  # convert the list into NumPy array
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # give as input to the model the set of windows X_test in order to returns the corresponding decoded windows.
    X_test_pred = model.predict(X_test)
    test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)  # find the losses

    # save again the first half - WINDOW_SIZE of the curve in order to make the plot
    test = dataset_total.iloc[:, dataset_total.shape[1] - dataset_test.shape[1] - WINDOW_SIZE:].T
    test_score_df = pd.DataFrame(index=test[WINDOW_SIZE:].index)
    test_score_df['loss'] = test_mae_loss
    test_score_df['threshold'] = THRESHOLD
    test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold  # spot the anomalies, based on the given threshold
    test_score_df['close'] = test[WINDOW_SIZE:]

    result_size = X_test.shape[0]
    time = list(range(1, result_size+1))

    # save the anomalies in a list
    anomalies = test_score_df[test_score_df.anomaly == True]
    anomalies_indexes = anomalies.index-test_score_df.index[0]
    anomalies_indexes = anomalies_indexes.tolist()
    anomalies_times = [time[i] for i in anomalies_indexes]

    f = plt.figure()
    f.set_figwidth(20)
    f.set_figheight(5)
    # plot the real curve
    plt.plot(time, test_score_df.close, color='red', label='Real Curve')
    # above the real curve plot note the points that anomalies was found
    plt.scatter(anomalies_times, anomalies.close, color='blue', label='Anomalies')
    plt.xticks(np.arange(0, result_size, 50), rotation=70)
    plt.title('Anomalies of curve: '+str(dataset.index[curve]))
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
plt.show()
