import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.layers import TimeDistributed

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

# %%
tf.random.set_seed(10)
BATCH_SIZE = 128
BUFFER_SIZE = 10000
EPOCH = 1000

# %%
past_history = 4 * 24 * 10
future_target = 4 * 24
STEP = 4
SHIFT_STEP = 1

# %%
def root_mean_squared_error_loss(y_true, y_pred):
    return tf.keras.backend.sqrt(tf.keras.losses.MSE(y_true, y_pred))


tf.random.set_seed(10)
df = pd.read_csv("../../data/datefrom1st.csv")
df.index = df.datetime
df = df.drop(
    ["Unnamed: 0", 'datetime', 'percipitation', 'air_pressure', 'sea_level_pressure',
     'wind_degree'], axis=1)
df["difference"] = df.astype('int32')
df['shift1'] = df['result'].shift(-SHIFT_STEP)
df['shift2'] = df['result'].shift(-(SHIFT_STEP+1))
df['shift3'] = df['result'].shift(-(SHIFT_STEP+2))


# %%
df.corr()
# %%

TRAIN_SPLIT = int(len(df.index) * 0.8)
scaler = MinMaxScaler().fit(df)
values = scaler.transform(df)
save = values
print(values)
# %%

train = values[:TRAIN_SPLIT, :]
test = values[TRAIN_SPLIT:, :]
# split into input and outputs
train_X, train_y = train[:, 1:], train[:, 0]
test_X, test_y = test[:, 1:], test[:, 0]
# %%

train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


# %%
def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()

    plt.show()


# %%

def show_plot(plot_data, delta, title):
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']
    time_steps = create_time_steps(plot_data[0].shape[0])
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10,
                     label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future + 5) * 2])
    plt.xlabel('Time-Step')
    return plt


def create_time_steps(length):
    return list(range(-length, 0))


# %%

def multi_step_plot(history, true_future, prediction):
    plt.figure(figsize=(12, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)

    plt.plot(num_in, np.array(history[:, 1]), label='History')
    plt.plot(np.arange(num_out) / STEP, np.array(true_future), 'bo',
             label='True Future')
    if prediction.any():
        plt.plot(np.arange(num_out) / STEP, np.array(prediction), 'ro',
                 label='Predicted Future')
    plt.legend(loc='upper left')
    plt.show()
    plt.savefig("output.png")


# %%

def fit_by_batch(X, y, batch_size):
    n_batches_for_epoch = X.shape[0]//batch_size
    for i in range(n_batches_for_epoch):
        index_batch = range(X.shape[0])[batch_size*i:batch_size*(i+1)]
        X_batch =X[index_batch][0].toarray()[0] #from sparse to array
        X_batch=X_batch.reshape(1,X_batch.shape[0],1 ) # to 3d array
        y_batch = y[index_batch,][0]
        yield(np.array(X_batch),y_batch)


import tensorflow as tf
from tensorflow.keras.layers import RepeatVector

multi_step_model = tf.keras.models.Sequential()
multi_step_model.add(tf.keras.layers.LSTM(train_X.shape[1], activation='relu', return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
multi_step_model.add(tf.keras.layers.LSTM(200, activation='relu', return_sequences=True))
multi_step_model.add(tf.keras.layers.LSTM(1, activation='relu'))
multi_step_model.add(tf.keras.layers.RepeatVector(train_X.shape[1]))
multi_step_model.add(tf.keras.layers.LSTM(train_X.shape[1], activation='relu', return_sequences=True))
multi_step_model.add(tf.keras.layers.LSTM(200, activation='relu', return_sequences=True))
multi_step_model.add(TimeDistributed(tf.keras.layers.Dense(1)))


# %%
multi_step_model.summary()

# %%
from matplotlib import pyplot
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.python.client import device_lib

EVALUATION_INTERVAL = 200


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def get_gpu_num():
    return len(get_available_gpus())


path_checkpoint = '../23_checkpoint.keras'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      monitor='val_loss',
                                      verbose=1,
                                      save_weights_only=True,
                                      save_best_only=True)

callback_early_stopping = EarlyStopping(monitor='val_loss', patience=80, verbose=1)

callback_tensorboard = TensorBoard(log_dir='../23_logs/', histogram_freq=0, write_graph=False)
callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, min_lr=1e-5, patience=0,verbose=1)
callbacks = [callback_early_stopping, callback_checkpoint, callback_tensorboard, callback_reduce_lr]

print(f"[+] Available GPUs")
print(get_available_gpus())

if get_gpu_num() < 2:
    print(f"[+] Available multiple GPU not found... Just use CPU! XD")
else:
    print(f"[+] {get_gpu_num()} GPUs found! Setting to GPU model...")
    multi_step_model = multi_gpu_model(multi_step_model, gpus=get_gpu_num())

multi_step_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')

history = multi_step_model.fit(train_X, train_y, epochs=EPOCH, batch_size=32 * 8, validation_data=(test_X, test_y),
                               verbose=2, shuffle=True, callbacks=callbacks)

try:
    multi_step_model.load_weights(path_checkpoint)
except Exception as error:
    print("Error trying to load checkpoint.")
    print(error)

# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
pyplot.savefig("test.png")

# %%

from numpy import concatenate
from math import sqrt
from sklearn.metrics import mean_squared_error

# make a prediction
yhat = multi_step_model.predict(test_X)[:, :, 0]
print(yhat)
# make a prediction
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.6f' % rmse)
print('Test MAE: %.6f' % mean_squared_error(inv_y, inv_yhat))
print('Test nMAE: %.6f' % (mean_squared_error(inv_y, inv_yhat) / 7028))

pyplot.plot([x for x in range(1000)], inv_y[-1000:], 'b', label='true')
pyplot.plot([x for x in range(1000)], inv_yhat[-1000:], 'r', label='pred')
pyplot.legend(loc='upper left')
pyplot.savefig("out.png")