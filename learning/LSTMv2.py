#%%



# %%

import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

# %%
tf.random.set_seed(10)
BATCH_SIZE = 128
BUFFER_SIZE = 10000
EPOCH=200
# %%

def root_mean_squared_error_loss(y_true, y_pred):
    return tf.keras.backend.sqrt(tf.keras.losses.MSE(y_true, y_pred))



tf.random.set_seed(10)
df = pd.read_csv("../data/datefrom1st.csv")
df.index = df.datetime
df = df.drop(
    ["temperature", "difference", "Unnamed: 0", 'datetime', 'percipitation', 'air_pressure', 'sea_level_pressure',
     'wind_degree'], axis=1)


#%%
df.corr()
#%%

TRAIN_SPLIT = int(len(df.index) * 0.8)
scaler = MinMaxScaler().fit(df)
values = scaler.transform(df)
save = values
print(values)
#%%

train = values[:TRAIN_SPLIT, :]
test = values[TRAIN_SPLIT:, :]
# split into input and outputs
train_X, train_y = train[:, 1:], train[:, 0]
test_X, test_y = test[:, 1:], test[:, 0]
# %%

train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)



#%%

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
import tensorflow as tf
from tensorflow.keras.layers import RepeatVector

multi_step_model = tf.keras.models.Sequential()
multi_step_model.add(tf.keras.layers.LSTM(200,activation="relu",
                                              return_sequences=True,
                                              input_shape=(train_X.shape[1], train_X.shape[2])))
multi_step_model.add(tf.keras.layers.Dense(1))

# %%
multi_step_model.summary()

# %%
from matplotlib import pyplot
EVALUATION_INTERVAL = 200
from tensorflow.keras.utils import multi_gpu_model

multi_step_model = multi_gpu_model(multi_step_model, gpus=8)

multi_step_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')

history = multi_step_model.fit(train_X, train_y, epochs=EPOCH, batch_size=32*4, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
pyplot.savefig("test.png")

#%%
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
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.6f' % rmse)
print('Test MAE: %.6f' % mean_squared_error(inv_y, inv_yhat))
print('Test nMAE: %.6f' % (mean_squared_error(inv_y, inv_yhat)/7028))

pyplot.plot([x for x in range (1000)], inv_y[-1000:], 'b', label='true')
pyplot.plot([x for x in range (1000)], inv_yhat[-1000:], 'r', label = 'pred')
pyplot.legend(loc='upper left')
pyplot.savefig("out.png")

