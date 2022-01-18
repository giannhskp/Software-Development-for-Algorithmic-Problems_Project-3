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


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
argv = sys.argv[1:]
data_loc = ''
query_loc = ''
out_data_loc = ''
out_query_loc = ''
NUMBER_OF_SAMPLE_CURVES = 4
for i, arg in enumerate(argv):
    if arg == "-d":
        data_loc = argv[i+1]
    elif arg == "-q":
        query_loc = argv[i+1]
    elif arg == "-od":
        out_data_loc = argv[i+1]
    elif arg == "-oq":
        out_query_loc = argv[i+1]
    elif arg == "-n":
        NUMBER_OF_SAMPLE_CURVES = int(argv[i+1])
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

TRAIN_NEW_MODEL = False
encoder_loc = 'models/part3/part3_encoder.h5'
autoencoder_loc = 'models/part3/part3_autoencoder.h5'

if TRAIN_NEW_MODEL:
    print('Training new model...')
    INPUT_SIZE = dataset.shape[0]
    SERIES_LENGTH = dataset.shape[1]-1
    TRAIN_LENGTH = math.floor(19*SERIES_LENGTH/20)
    EPOCHS = 20
    BATCH_SIZE = 512
    LEARNING_RATE = 0.0001
    WINDOW_SIZE = 10
    SHUFFLE_TRAIN_DATA = True
    TRAIN_CURVES = list(range(INPUT_SIZE))

    training_set = dataset.iloc[:, 1:TRAIN_LENGTH+1].values
    test_set = dataset.iloc[:, TRAIN_LENGTH+1:TRAIN_LENGTH*2+1].values

    scaler = MinMaxScaler()

    x_test_list = []
    x_train_list = []
    for curve in TRAIN_CURVES:
        dataset_train = dataset.iloc[curve:curve+1, 1:TRAIN_LENGTH+1].values
        dataset_train = dataset_train.reshape(-1, 1)
        dataset_test = dataset.iloc[curve:curve+1, TRAIN_LENGTH+1:].values
        dataset_test = dataset_test.reshape(-1, 1)

        df = pd.DataFrame(np.array(dataset_train)[:, 0], columns=['price'])
        x_train_list = x_train_list + ([scaler.fit_transform(df['price'].values[i-WINDOW_SIZE:i].reshape(-1, 1)) for i in (range(WINDOW_SIZE, len(df['price'])))])

        df_test = pd.DataFrame(np.array(dataset_test)[:, 0], columns=['price'])
        x_test_list = x_test_list + ([scaler.fit_transform(df_test['price'].values[i-WINDOW_SIZE:i].reshape(-1, 1)) for i in (range(WINDOW_SIZE, len(df_test['price'])))])
    x_train = np.array(x_train_list)
    x_test = np.array(x_test_list)

    input_window = Input(shape=(WINDOW_SIZE, 1))
    x = Conv1D(16, 3, activation="relu", padding="same")(input_window)  # 10 dims
    x = MaxPooling1D(2, padding="same")(x)  # 5 dims
    x = Conv1D(1, 3, activation="relu", padding="same")(x)  # 5 dims
    encoded = MaxPooling1D(2, padding="same")(x)  # 3 dims

    encoder = Model(input_window, encoded)

    # 3 dimensions in the encoded layer

    x = Conv1D(1, 3, activation="relu", padding="same")(encoded)  # 3 dims
    x = UpSampling1D(2)(x)  # 6 dims
    x = Conv1D(16, 2, activation='relu')(x)  # 5 dims
    x = UpSampling1D(2)(x)  # 10 dims
    decoded = Conv1D(1, 3, activation='sigmoid', padding='same')(x)  # 10 dims
    autoencoder = Model(input_window, decoded)
    autoencoder.summary()

    optimizer = Adam(learning_rate=LEARNING_RATE)
    autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy')
    history = autoencoder.fit(x_train, x_train,
                              epochs=EPOCHS,
                              batch_size=BATCH_SIZE,
                              shuffle=SHUFFLE_TRAIN_DATA,
                              validation_data=(x_test, x_test)
                              )
    autoencoder.save('part3_autoencoder.h5')
    encoder.save('part3_encoder.h5')
    print('Finished training. New model was saved')
else:
    encoder = keras.models.load_model(encoder_loc, compile=False)
    autoencoder = keras.models.load_model(autoencoder_loc)

WINDOW_SIZE = 10

query_result_list = []
for curve in range(queryset.shape[0]):
    original_curve = queryset.iloc[curve:curve+1, :].values
    original_curve = original_curve.reshape(-1, 1)
    df = pd.DataFrame(np.array(original_curve)[:, 0], columns=['price'])
    loop_range = range(0, math.floor((len(df['price']))/WINDOW_SIZE))
    scaler_list = []
    transformed_list = []
    for i in loop_range:
        temp_scaler = MinMaxScaler()
        transformed = np.array(temp_scaler.fit_transform(df['price'].values[i*WINDOW_SIZE:(i+1)*WINDOW_SIZE].reshape(-1, 1)))

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
print('Query file ready')


dataset_result_list = []
for curve in range(dataset.shape[0]):
    original_curve = dataset.iloc[curve:curve+1, :].values
    original_curve = original_curve.reshape(-1, 1)
    df = pd.DataFrame(np.array(original_curve)[:, 0], columns=['price'])
    loop_range = range(0, math.floor((len(df['price']))/WINDOW_SIZE))
    scaler_list = []
    transformed_list = []
    for i in loop_range:
        temp_scaler = MinMaxScaler()
        transformed = np.array(temp_scaler.fit_transform(df['price'].values[i*WINDOW_SIZE:(i+1)*WINDOW_SIZE].reshape(-1, 1)))

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
print('Input file ready')


def plot_time_series(plot_ax, y, x, title):
    x = list(range(1, x+1))
    plot_ax.plot(x, y, 'r', label='value')
    plot_ax.legend(loc='best')
    plot_ax.set_xlabel('Time')
    plot_ax.set_ylabel('Value')
    plot_ax.title.set_text(title)


DATASET_SIZE = dataset.shape[0]
print('Plotting ', NUMBER_OF_SAMPLE_CURVES, ' random original/compressed curves for result comparison')
for curve in list(random.sample(range(0, DATASET_SIZE), NUMBER_OF_SAMPLE_CURVES)):
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
    # tranformed_full = autoencoder.predict(x_transform)

    unscaled_list = []
    for i, x in enumerate(tranformed):
        t_scale = scaler_list[i]
        inversed = np.array(t_scale.inverse_transform(x))
        unscaled_list.append(inversed)
    unscaled = np.array(unscaled_list)
    unscaled = unscaled.reshape(-1, 1)

    fig, ax = plt.subplots(2, 1, figsize=(10, 5))
    fig.tight_layout()

    plot_time_series(ax[0], original_curve, original_curve.shape[0], 'Original Curve: '+dataset.index[curve])
    plot_time_series(ax[1], unscaled, unscaled.shape[0], 'Compressed Curve: '+dataset.index[curve])
plt.show()
