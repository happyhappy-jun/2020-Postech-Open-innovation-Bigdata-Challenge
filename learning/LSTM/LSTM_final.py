
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

# %%
tf.random.set_seed(10)
BATCH_SIZE = 32
BUFFER_SIZE = 10000
EPOCH = 1000
DROPOUT = 0.2
# %%
past_history = 4 * 24 * 10
future_target = 4 * 24
STEP = 4
SHIFT_STEP = 1

# %%
def root_mean_squared_error_loss(y_true, y_pred):
    return tf.keras.backend.sqrt(tf.keras.losses.MSE(y_true, y_pred))

def xy_split(d, scale, y=True):
    d_trans = scale.transform(d)
    if y:
        d_x = d_trans[:,1:]
        d_y = d_trans[:, 0]
        d_x = d_x.reshape((d_x.shape[0], 1, d_x.shape[1]))
        return d_x, d_y
    if not y:
        d_x = d_trans[:,1:]
        d_x = d_x.reshape((d_x.shape[0], 1, d_x.shape[1]))
        return d_x


tf.random.set_seed(42)
raw_df = pd.read_csv("data/datefrom1st.csv")
raw_df.index = raw_df.datetime

df = raw_df
df = df.drop(
    [ "Unnamed: 0", 'datetime', 'percipitation', 'air_pressure', 'sea_level_pressure',
     'wind_degree'], axis=1)
df["difference"] = df.astype('int32')
df.loc[df["solar_radiation"]<0 , "solar_radiation"] = 0
df.loc[df["solar_intensity"]<0 , "solar_intensity"] = 0

final_test = df.loc["2020-05-31 00:00:00":"2020-05-31 23:45:00"]
df.drop(df.loc[(df.index > '2020-01-31 00:00:00') & (df.index < '2020-02-01 00:00:00')].index, inplace=True)
df.drop(df.loc[(df.index > '2020-03-31 00:00:00') & (df.index < '2020-04-01 00:00:00')].index, inplace=True)
df.drop(df.loc[(df.index > '2020-05-31 00:00:00') & (df.index < '2020-06-01 00:00:00')].index, inplace=True)
df = df.fillna(0)


# %%

TRAIN_SPLIT = int(len(df.index) * 0.8)
scaler = MinMaxScaler().fit(df)
values = df.values


train = values[:TRAIN_SPLIT, :]
test = values[TRAIN_SPLIT:, :]


# split into input and outputs
train_X, train_y = xy_split(train, scaler)
test_X, test_y = xy_split(test, scaler)
final_test_X, final_test_y = xy_split(final_test, scaler)

# %%x
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


print(train_X.shape)
import tensorflow as tf

# %%
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.python.client import device_lib


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

callback_early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1)

callback_tensorboard = TensorBoard(log_dir='../23_logs/', histogram_freq=0, write_graph=False)
callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, min_lr=1e-7, patience=10,verbose=1)
callbacks = [callback_early_stopping, callback_checkpoint, callback_tensorboard, callback_reduce_lr]





multi_step_model = tf.keras.models.Sequential()
multi_step_model.add(tf.keras.layers.GRU(300, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
multi_step_model.add(tf.keras.layers.GRU(300, return_sequences=True))
multi_step_model.add(tf.keras.layers.Dense(1))





print(f"[+] Available GPUs")
print(get_available_gpus())

if get_gpu_num() < 2:
    print(f"[+] Available multiple GPU not found... Just use CPU! XD")
else:
    print(f"[+] {get_gpu_num()} GPUs found! Setting to GPU model...")
    multi_step_model = multi_gpu_model(multi_step_model, gpus=get_gpu_num())

multi_step_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')

history = multi_step_model.fit(train_X, train_y, epochs=EPOCH, batch_size=BATCH_SIZE, validation_data=(test_X, test_y),
                               verbose=2, shuffle=True, callbacks=callbacks)

try:
    multi_step_model.load_weights(path_checkpoint)
except Exception as error:
    print("Error trying to load checkpoint.")

from numpy import concatenate
from math import sqrt
from sklearn.metrics import mean_squared_error


def make_prediction(model, X, y, plot_name):
    yhat = model.predict(X)[:, :, 0]
    X_revert = X.reshape((X.shape[0], X.shape[2]))
    inv_yhat = concatenate((yhat, X_revert), axis = 1)
    inv_xyhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_xyhat[:, 0]
    y_revert = y.reshape((len(y), 1))
    inv_y1 = concatenate((y_revert, X_revert), axis = 1)
    print("after concate: {}".format(inv_y1.shape))
    inv_y1 = scaler.inverse_transform(inv_y1)
    inv_y = inv_y1[:, 0]
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.6f' % rmse)
    print('Test MAE: %.6f' % mean_squared_error(inv_y, inv_yhat))


    fig = plt.figure(figsize=(20,6))
    plt.plot(inv_y, 'b', label = 'true')
    plt.plot(inv_yhat, 'g', label = 'pred')
    plt.legend()
    plt.savefig(plot_name)

make_prediction(multi_step_model, final_test_X, final_test_y, "final_test.png")
