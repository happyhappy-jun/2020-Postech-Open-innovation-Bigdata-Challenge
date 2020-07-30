import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler


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
def xy_split(d, scale, y=True):
    d_trans = scale.transform(d)
    if y:
        d_x = d_trans[:, 1:]
        d_y = d_trans[:, 0]
        d_x = d_x.reshape((d_x.shape[0], 1, d_x.shape[1]))
        return d_x, d_y
    if not y:
        d_x = d_trans[:, 1:]
        d_x = d_x.reshape((d_x.shape[0], 1, d_x.shape[1]))
        return d_x


tf.random.set_seed(42)
raw_df = pd.read_csv("../../data/datefrom1st.csv")
raw_df.index = raw_df.datetime

df = raw_df
df = df.drop(
    ["Unnamed: 0", 'datetime', 'percipitation', 'air_pressure', 'sea_level_pressure',
     'wind_degree'], axis=1)
df["difference"] = df["difference"].astype('int32')
df.loc[df["solar_radiation"] < 0, "solar_radiation"] = 0
df.loc[df["solar_intensity"] < 0, "solar_intensity"] = 0
df = df.fillna(0)
scaler = MinMaxScaler().fit(df)

final_test = df.loc["2020-05-24 00:00:00": "2020-05-31 00:00:00"]
final_test_5 = df.loc["2020-05-31 00:00:00":"2020-05-31 23:45:00"]
final_test_3 = df.loc["2020-03-31 00:00:00":"2020-03-31 23:45:00"]
final_test_1 = df.loc["2020-01-31 00:00:00":"2020-01-31 23:45:00"]
df.drop(df.loc[(df.index > '2020-01-31 00:00:00') & (df.index < '2020-02-01 00:00:00')].index, inplace=True)
df.drop(df.loc[(df.index > '2020-03-31 00:00:00') & (df.index < '2020-04-01 00:00:00')].index, inplace=True)
df.drop(df.loc[(df.index > '2020-05-31 00:00:00') & (df.index < '2020-06-01 00:00:00')].index, inplace=True)

# %%

TRAIN_SPLIT = int(len(df.index) * 0.8)
values = df.values

train = values[:TRAIN_SPLIT, :]
test = values[TRAIN_SPLIT:, :]

# split into input and outputs
train_X, train_y = xy_split(train, scaler)
test_X, test_y = xy_split(test, scaler)
final_test_X, final_test_y = xy_split(final_test, scaler)
final_test_X_1, final_test_y_1 = xy_split(final_test_1, scaler)
final_test_X_3, final_test_y_3 = xy_split(final_test_3, scaler)
final_test_X_5, final_test_y_5 = xy_split(final_test_5, scaler)

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

callback_early_stopping = EarlyStopping(monitor='val_loss', patience=100, verbose=1)

callback_tensorboard = TensorBoard(log_dir='../23_logs/', histogram_freq=0, write_graph=False)
callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, min_lr=1e-9, patience=30, verbose=1)
callbacks = [callback_early_stopping, callback_checkpoint, callback_tensorboard, callback_reduce_lr]

# Keras Tuner

###########################################################################################
# def build_model(hp):
#    multi_step_model = tf.keras.models.Sequential()

#    multi_step_model.add(tf.keras.layers.GRU(units=hp.Int('units', min_value = 100, max_value=400, step=50),
#    activation=hp.Choice('activation', ["relu", "tanh"]), return_sequences=True,
#    input_shape=(train_X.shape[1], train_X.shape[2])))
#    multi_step_model.add(tf.keras.layers.GRU(units=hp.Int('units', min_value=100, max_value=400, step=50),
#                                             activation=hp.Choice('activation', ["relu", "tanh"]),
#                                             return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))

# multi_step_model.add(tf.keras.layers.GRU(32, activation = "relu", return_sequences=True,
# input_shape=(train_X.shape[1], train_X.shape[2])))
# multi_step_model.add(tf.keras.layers.GRU(32, activation = "relu",  return_sequences=True))
# multi_step_model.add(tf.keras.layers.GRU(32, activation = "relu",  return_sequences=True))
# multi_step_model.add(tf.keras.layers.GRU(32, activation = "relu",  return_sequences=True))
#    multi_step_model.add(tf.keras.layers.Dense(1))
#    multi_step_model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate',
#    values=[1e-2, 1e-3, 1e-4])), loss='mse')
#    return multi_step_model


# tuner = RandomSearch(
#     build_model,
#     objective='val_loss',
#     max_trials=5,
#     executions_per_trial=1,
#     directory='H-tuning')
# 
# print(tuner.search_space_summary())
# tuner.search(train_X, train_y,epochs=EPOCH, batch_size=BATCH_SIZE, validation_data=(test_X, test_y),
#                                verbose=2, shuffle=True, callbacks=callbacks)
# print(tuner.results_summary())
##############################################################################

multi_step_model = tf.keras.models.Sequential()
multi_step_model.add(tf.keras.layers.LSTM(350, activation="relu", return_sequences=True,
                                         input_shape=(train_X.shape[1], train_X.shape[2])))
multi_step_model.add(tf.keras.layers.LSTM(350, activation="relu", return_sequences=True))
multi_step_model.add(tf.keras.layers.Dense(1))
multi_step_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='mse')

print(f"[+] Available GPUs")
print(get_available_gpus())

if get_gpu_num() < 2:
    print(f"[+] Available multiple GPU not found... Just use CPU! XD")
else:
    print(f"[+] {get_gpu_num()} GPUs found! Setting to GPU model...")
    multi_step_model = multi_gpu_model(multi_step_model, gpus=get_gpu_num())

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
    inv_yhat = concatenate((yhat, X_revert), axis=1)
    inv_xyhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_xyhat[:, 0]
    np.savetxt("output/ " + plot_name + ".csv", inv_yhat, delimiter = ",")
    y_revert = y.reshape((len(y), 1))
    inv_y1 = concatenate((y_revert, X_revert), axis=1)
    print("after concate: {}".format(inv_y1.shape))
    inv_y1 = scaler.inverse_transform(inv_y1)
    inv_y = inv_y1[:, 0]
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.6f' % rmse)
    print('Test MAE: %.6f' % mean_squared_error(inv_y, inv_yhat))

    fig = plt.figure(figsize=(20, 6))
    plt.plot(inv_y, 'b', label='true')
    plt.plot(inv_yhat, 'g', label='pred')
    plt.legend()
    plt.savefig("output/ " + plot_name + ".png")

    make_prediction(multi_step_model, final_test_X_1, final_test_y_1, "final_1")
    make_prediction(multi_step_model, final_test_X_3, final_test_y_3, "final_3")
    make_prediction(multi_step_model, final_test_X_5, final_test_y_5, "final_5")
    print(multi_step_model.summary())

# %%
# 2017 8 3
y = 17
date = 3
base = pd.read_csv("../../data/future/SURFACE_ASOS_131_MI_2017-08_2017-08_2017.csv", encoding='euc-kr').fillna(0)

base = base[['일시', '기온(°C)', '풍속(m/s)', '습도(%)', '일사(MJ/m^2)', '일조(Sec)']]
base = base.rename(columns={'일시': 'datetime', '기온(°C)': "temperature", '풍속(m/s)': "wind_speed", '습도(%)': "humidity",
                            "일조(Sec)": "solar_radiation", '일사(MJ/m^2)': "solar_intensity"
                            })
base.datetime = pd.to_datetime(base.datetime, infer_datetime_format=True)
base["difference"] = base["datetime"].sub(pd.to_datetime("2017-01-01", infer_datetime_format=True),
                                          axis=0) / np.timedelta64(1, 'D')

base = base.set_index('datetime')
base = base[f"20{y}-08-0{date} 00:00":f"20{y}-08-0{date} 23:59"]

# %%
out = pd.DataFrame()
period_min = 15
out["result"] = 0
out["temperature"] = base["temperature"].resample(f'{str(period_min)}T').mean()
out["wind_speed"] = base["wind_speed"].resample(f'{str(period_min)}T').mean()
out["humidity"] = base["humidity"].resample(f'{str(period_min)}T').mean()
out["solar_intensity"] = base["solar_intensity"].resample(f'{str(period_min)}T').mean().diff()
out["solar_radiation"] = base["solar_radiation"].resample(f'{str(period_min)}T').mean().diff()
out["date"] = 31
out["month"] = 7
out["hour"] = pd.DatetimeIndex(out.index).hour
out["difference"] = base["difference"]
out.loc[out["solar_intensity"] < 0, "solar_intensity"] = 0
out.loc[out["solar_radiation"] < 0, "solar_radiation"] = 0
out = out.fillna(0)
out_X, out_y = xy_split(out, scaler)
make_prediction(multi_step_model, out_X, out_y, "final_7")
