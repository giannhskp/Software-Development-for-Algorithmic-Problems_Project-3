import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
import keras
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Input
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import UpSampling1D
from keras.models import Model
from tensorflow.keras.optimizers import Adam


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
argv = sys.argv[1:]  # get parameters from command line
data_loc = ''  # location of dataset file given by the user
query_loc = ''  # location of query file given by the user
out_data_loc = ''  # location of output dataset file given by the user
out_query_loc = ''  # location of output query file given by the user
NUMBER_OF_SAMPLE_CURVES = 4  # number of compressed time series to be presented (default = 4)
for i, arg in enumerate(argv):  # read parameters
    if arg == "-d":
        data_loc = argv[i+1]
    elif arg == "-q":
        query_loc = argv[i+1]
    elif arg == "-od":
        out_data_loc = argv[i+1]
    elif arg == "-oq":
        out_query_loc = argv[i+1]
    elif arg == "-n":
        # optional parameter to set how many compressed time series will be presented
        NUMBER_OF_SAMPLE_CURVES = int(argv[i+1])
if data_loc == '' or query_loc == '' or out_data_loc == '' or out_query_loc == '':
    print('Wrong Arguments! \n Usage: $python reduce.py –d  <dataset> -q <queryset> -od <output_dataset_file> -oq <output_query_file>')
    sys.exit(2)
else:
    print('Dataset file is ', data_loc)
    print('Query file is ', query_loc)
    print('Dataset output file is ', out_data_loc)
    print('Query output is ', out_query_loc)

# read dataset file and store the data into a dataframe
dataset = pd.read_csv(data_loc, index_col=0, sep='\t', header=None)
# read query file and store the data into a dataframe
queryset = pd.read_csv(query_loc, index_col=0, sep='\t', header=None)

TRAIN_NEW_MODEL = False  # don't train a new model, load the saved one
# if TRAIN_NEW_MODEL is set to True a new model is trained using the given dataset file
encoder_loc = 'models/part3/part3_encoder.h5'  # location of the pre-trained encoder file
autoencoder_loc = 'models/part3/part3_autoencoder.h5'  # location of the pre-trained encoder file

if TRAIN_NEW_MODEL:
    print('Training new model...')
    INPUT_SIZE = dataset.shape[0]  # the number of time series of the dataset
    SERIES_LENGTH = dataset.shape[1]-1  # the length of each time series
    TRAIN_LENGTH = math.floor(19*SERIES_LENGTH/20)  # the length of each time series that will be used in train set (95%)
    # hyper parameters
    EPOCHS = 20
    BATCH_SIZE = 512
    LEARNING_RATE = 0.0001
    WINDOW_SIZE = 10
    SHUFFLE_TRAIN_DATA = True
    # the curves that will be used for the training
    TRAIN_CURVES = list(range(INPUT_SIZE))  # all the dataset curves will be used
    # create train set, using the first 95% of each curve
    training_set = dataset.iloc[:, 1:TRAIN_LENGTH+1].values
    # create test set, using the rest 5% of each curve
    test_set = dataset.iloc[:, TRAIN_LENGTH+1:TRAIN_LENGTH*2+1].values

    scaler = MinMaxScaler()  # initialize scaler

    x_test_list = []
    x_train_list = []
    for curve in TRAIN_CURVES:  # for every curve
        # get the first 95% of the curve in order to use it in the train set
        dataset_train = dataset.iloc[curve:curve+1, 1:TRAIN_LENGTH+1].values
        dataset_train = dataset_train.reshape(-1, 1)
        # get the rest 5% of the curve in order to use it in the test set
        dataset_test = dataset.iloc[curve:curve+1, TRAIN_LENGTH+1:].values
        dataset_test = dataset_test.reshape(-1, 1)
        # create a dataframe using the train values of this curve
        df = pd.DataFrame(np.array(dataset_train)[:, 0], columns=['price'])
        # split the train values into overlapping windows, scale each window seperatly, and store all the scaled window in a list
        x_train_list = x_train_list + ([scaler.fit_transform(df['price'].values[i-WINDOW_SIZE:i].reshape(-1, 1)) for i in (range(WINDOW_SIZE, len(df['price'])))])
        # create a dataframe using the test values of this curve
        df_test = pd.DataFrame(np.array(dataset_test)[:, 0], columns=['price'])
        # split the test values into overlapping windows, scale each window seperatly, and store all the scaled window in a list
        x_test_list = x_test_list + ([scaler.fit_transform(df_test['price'].values[i-WINDOW_SIZE:i].reshape(-1, 1)) for i in (range(WINDOW_SIZE, len(df_test['price'])))])
    x_train = np.array(x_train_list)  # convert the list into a numpy array
    x_test = np.array(x_test_list)  # convert the list into a numpy array

    # MODEL DEFINITION ##########################
    input_window = Input(shape=(WINDOW_SIZE, 1))  # input layer - output shape: (10,1)
    x = Conv1D(16, 3, activation="relu", padding="same")(input_window)  # 10 dims - output shape: (10,16)
    x = MaxPooling1D(2, padding="same")(x)  # lower the dimension to 5 - output shape: (5,16)
    x = Conv1D(1, 3, activation="relu", padding="same")(x)  # 5 dims - output shape: (5,1)
    encoded = MaxPooling1D(2, padding="same")(x)  # lower the dimension to 3 - output shape: (3,1)

    # create the encoder model that will be used for the time series compression
    encoder = Model(input_window, encoded)
    # 3 dimensions in the encoded layer

    x = Conv1D(1, 3, activation="relu", padding="same")(encoded)  # 3 dims - output shape: (3,1)
    x = UpSampling1D(2)(x)  # double the dimension to 6 - output shape: (6,1)
    x = Conv1D(16, 2, activation='relu')(x)  # 5 dims - output shape: (5,16)
    x = UpSampling1D(2)(x)  # double the dimension to 10 - output shape: (10,16)
    decoded = Conv1D(1, 3, activation='sigmoid', padding='same')(x)  # 10 dims - output shape: (10,1)
    # create the complete model of the auto-encoder
    autoencoder = Model(input_window, decoded)
    autoencoder.summary()  # print the model structure

    optimizer = Adam(learning_rate=LEARNING_RATE)  # initialize the adam optimizer with the assinged learning rate
    # compile the model. Adam optimizer and binary_crossentropy loss function will be used
    autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy')
    # start the training of the model
    # the test set is used as validation set
    history = autoencoder.fit(x_train, x_train,
                              epochs=EPOCHS,
                              batch_size=BATCH_SIZE,
                              shuffle=SHUFFLE_TRAIN_DATA,
                              validation_data=(x_test, x_test)
                              )
    # save the model, both encoder and autoencoder
    autoencoder.save('part3_autoencoder.h5')
    encoder.save('part3_encoder.h5')
    print('Finished training. New model was saved')
else:
    # load the saved pre-trained model
    encoder = keras.models.load_model(encoder_loc, compile=False)
    # autoencoder is actualy not used
    autoencoder = keras.models.load_model(autoencoder_loc)

WINDOW_SIZE = 10

query_result_list = []
for curve in range(queryset.shape[0]):  # for every time series in the query file
    original_curve = queryset.iloc[curve:curve+1, :].values  # get the time series
    original_curve = original_curve.reshape(-1, 1)
    # create a dataframe containing the values of the time series
    df = pd.DataFrame(np.array(original_curve)[:, 0], columns=['price'])
    # compute how many windows will be created (series_length/window_size)
    loop_range = range(0, math.floor((len(df['price']))/WINDOW_SIZE))
    scaler_list = []
    transformed_list = []
    # Loop that creates all the windows
    # Windows are NOT overlapping. Every window is scaled seperatly
    for i in loop_range:  # for every window
        temp_scaler = MinMaxScaler()  # initialize the scaler
        # get the window and scale it
        transformed = np.array(temp_scaler.fit_transform(df['price'].values[i*WINDOW_SIZE:(i+1)*WINDOW_SIZE].reshape(-1, 1)))
        # store the scaler in order to use him for the inverse_transform of the corresponding output
        scaler_list.append(temp_scaler)
        transformed_list.append(transformed)  # add the scaled window to the list
    # finally x_transform contains all the scaled windows
    x_transform = np.array(transformed_list)

    tranformed = encoder.predict(x_transform)  # use the encoder to produce the compressed windows

    unscaled_list = []
    for i, x in enumerate(tranformed):  # for every compressed window
        t_scale = scaler_list[i]  # get the scaler thas was used for the corresponding input window
        inversed = np.array(t_scale.inverse_transform(x))  # unscale the output window
        unscaled_list.append(inversed)
    # finally unscaled_list contains all the unscaled compressed windows
    unscaled = np.array(unscaled_list)
    unscaled = unscaled.reshape(-1, 1)
    query_result_list.append(unscaled)  # convert list to array

query_result = np.array(query_result_list)  # convert list to array
query_result = query_result.reshape(query_result.shape[0], query_result.shape[1])
# create dataframe that contains all the compressed time series
query_df = pd.DataFrame(data=query_result, index=queryset.index[:])
# create the output csv file from the dataframe
query_df.to_csv(out_query_loc, sep='\t', encoding='utf-8', header=False)
print('Query file ready')


dataset_result_list = []
for curve in range(dataset.shape[0]):  # for every time series in the dataset file
    original_curve = dataset.iloc[curve:curve+1, :].values  # get the time series
    original_curve = original_curve.reshape(-1, 1)
    # create a dataframe containing the values of the time series
    df = pd.DataFrame(np.array(original_curve)[:, 0], columns=['price'])
    # compute how many windows will be created (series_length/window_size)
    loop_range = range(0, math.floor((len(df['price']))/WINDOW_SIZE))
    scaler_list = []
    transformed_list = []
    # Loop that creates all the windows
    # Windows are NOT overlapping. Every window is scaled seperatly
    for i in loop_range:
        temp_scaler = MinMaxScaler()  # initialize the scaler
        transformed = np.array(temp_scaler.fit_transform(df['price'].values[i*WINDOW_SIZE:(i+1)*WINDOW_SIZE].reshape(-1, 1)))
        # store the scaler in order to use him for the inverse_transform of the corresponding output
        scaler_list.append(temp_scaler)
        transformed_list.append(transformed)  # add the scaled window to the list
    # finally x_transform contains all the scaled windows
    x_transform = np.array(transformed_list)

    tranformed = encoder.predict(x_transform)  # use the encoder to produce the compressed windows

    unscaled_list = []
    for i, x in enumerate(tranformed):  # for every compressed window
        t_scale = scaler_list[i]  # get the scaler thas was used for the corresponding input window
        inversed = np.array(t_scale.inverse_transform(x))  # unscale the output window
        unscaled_list.append(inversed)
    # finally unscaled_list contains all the unscaled compressed windows
    unscaled = np.array(unscaled_list)
    unscaled = unscaled.reshape(-1, 1)
    dataset_result_list.append(unscaled)  # convert list to array

dataset_result = np.array(dataset_result_list)  # convert list to array
dataset_result = dataset_result.reshape(dataset_result.shape[0], dataset_result.shape[1])
# create dataframe that contains all the compressed time series
dataset_df = pd.DataFrame(data=dataset_result, index=dataset.index[:])
# create the output csv file from the dataframe
dataset_df.to_csv(out_data_loc, sep='\t', encoding='utf-8', header=False)
print('Input file ready')


def plot_time_series(plot_ax, y, x, title):
    # function that plots a time series
    x = list(range(1, x+1))  # time vector (x)
    plot_ax.plot(x, y, 'r', label='value')
    plot_ax.legend(loc='best')
    plot_ax.set_xlabel('Time')
    plot_ax.set_ylabel('Value')
    plot_ax.title.set_text(title)


DATASET_SIZE = dataset.shape[0]
print('Plotting ', NUMBER_OF_SAMPLE_CURVES, ' random original/compressed curves for result comparison')
# select NUMBER_OF_SAMPLE_CURVES random time series from the dataset file
# for each time series, plot the original and the compressed
for curve in list(random.sample(range(0, DATASET_SIZE), NUMBER_OF_SAMPLE_CURVES)):
    # get the certain time series from the dataset
    original_curve = dataset.iloc[curve:curve+1, :].values
    original_curve = original_curve.reshape(-1, 1)
    # create a dataframe containing the values of the time series
    df = pd.DataFrame(np.array(original_curve)[:, 0], columns=['price'])
    # compute how many windows will be created (series_length/window_size)
    loop_range = range(0, math.floor((len(df['price'])-1)/WINDOW_SIZE))
    scaler_list = []
    transformed_list = []
    # Loop that creates all the windows
    # Windows are NOT overlapping. Every window is scaled seperatly
    for i in loop_range:  # for every window
        temp_scaler = MinMaxScaler()  # initialize the scaler
        # get the window and scale it
        transformed = np.array(temp_scaler.fit_transform(df['price'].values[i*WINDOW_SIZE+1:(i+1)*WINDOW_SIZE+1].reshape(-1, 1)))
        # store the scaler in order to use him for the inverse_transform of the corresponding output
        scaler_list.append(temp_scaler)
        transformed_list.append(transformed)  # add the scaled window to the list
    # finally x_transform contains all the scaled windows
    x_transform = np.array(transformed_list)

    tranformed = encoder.predict(x_transform)  # use the encoder to produce the compressed windows

    unscaled_list = []
    for i, x in enumerate(tranformed):  # for every compressed window
        t_scale = scaler_list[i]  # get the scaler thas was used for the corresponding input window
        inversed = np.array(t_scale.inverse_transform(x))  # unscale the output window
        unscaled_list.append(inversed)
    # finally unscaled_list contains all the unscaled compressed windows
    unscaled = np.array(unscaled_list)
    unscaled = unscaled.reshape(-1, 1)

    fig, ax = plt.subplots(2, 1, figsize=(10, 5))  # create a subplot
    fig.tight_layout()
    # plot the original curve in the first plot of the subplot
    plot_time_series(ax[0], original_curve, original_curve.shape[0], 'Original Curve: '+dataset.index[curve])
    # plot the compressed curve in the second plot of the subplot
    plot_time_series(ax[1], unscaled, unscaled.shape[0], 'Compressed Curve: '+dataset.index[curve])
plt.show()
