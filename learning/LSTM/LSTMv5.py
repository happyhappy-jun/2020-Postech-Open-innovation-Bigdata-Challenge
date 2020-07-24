# %%

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.backend import square, mean

# %%

tf.random.set_seed(10)
raw_df = pd.read_csv("../../data/datefrom1st_revised.csv")
raw_df.index = raw_df.datetime

df = raw_df

df = df.drop(
    ["Unnamed: 0", 'datetime', 'percipitation', 'air_pressure', 'sea_level_pressure',
     'wind_degree'], axis=1)
df["difference"] = df.astype('int32')

target_X = df.loc["2020-01-30 00:00:00":"2020-01-30 23:45:00"]

df.drop(df.loc[(df.index > '2020-01-31 00:00:00') & (df.index < '2020-02-01 00:00:00')].index, inplace=True)
df.drop(df.loc[(df.index > '2020-03-31 00:00:00') & (df.index < '2020-04-01 00:00:00')].index, inplace=True)
df.drop(df.loc[(df.index > '2020-05-31 00:00:00') & (df.index < '2020-06-01 00:00:00')].index, inplace=True)
df = df.fillna(0)
df = df.loc[:"2020-01-31 00:00:00"]
ult = df.loc["2020-01-24 00:00:00":]
TRAIN_SPLIT = int(len(df.index) * 0.8)
print(df.head())

dataset = df

shift_days = 1
STEP = 4
shift_steps = shift_days * 24 * 4  # Number of hours.
df_targets = df["result"].shift(-shift_steps)
df = df.drop("result", axis=1)
x_data = df.values[0:-shift_steps]
print(type(x_data))
print("Shape:", x_data.shape)
y_data = df_targets.values[:-shift_steps]
print(type(y_data))
print("Shape:", y_data.shape)

# %%

num_data = len(x_data)
num_data

train_split = 0.8

num_train = int(train_split * num_data)

x_train = x_data[0:num_train]
x_test = x_data[num_train:]

y_train = y_data[0:num_train]
y_test = y_data[num_train:]

num_x_signals = x_data.shape[1]
num_y_signals = 1
print("Min:", np.min(x_train))
print("Max:", np.max(x_train))

x_scaler = MinMaxScaler()

# %%

x_train_scaled = x_scaler.fit_transform(x_train)


# %%

print("Min:", np.min(x_train_scaled))
print("Max:", np.max(x_train_scaled))

x_test_scaled = x_scaler.transform(x_test)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_scaler = MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)
ult_x_scaled = x_scaler.transform(ult.iloc[:, 1:])
ult_y_test = ult.iloc[:,0].values.reshape(-1,1)
ult_y_scaled = y_scaler.transform(ult_y_test)

# %%

print(x_train_scaled.shape)
print(y_train_scaled.shape)


# %%

def batch_generator(batch_size, sequence_length):
    """
    Generator function for creating random batches of training-data.
    """

    # Infinite loop.
    while True:
        # Allocate a new array for the batch of input-signals.
        x_shape = (batch_size, sequence_length, num_x_signals)
        x_batch = np.zeros(shape=x_shape, dtype=np.float16)

        # Allocate a new array for the batch of output-signals.
        print((batch_size, sequence_length, num_y_signals))
        y_shape = (batch_size, sequence_length, num_y_signals)
        y_batch = np.zeros(shape=y_shape, dtype=np.float16)

        # Fill the batch with random sequences of data.
        for i in range(batch_size):
            # Get a random start-index.
            # This points somewhere into the training-data.
            idx = np.random.randint(num_train - sequence_length)

            # Copy the sequences of data starting at this index.
            x_batch[i] = x_train_scaled[idx:idx + sequence_length]
            y_batch[i] = y_train_scaled[idx:idx + sequence_length]

        yield (x_batch, y_batch)


# %%

batch_size = 128

# %%

sequence_length = 24 * 7 * 4 * 4
sequence_length

# %%

generator = batch_generator(batch_size=batch_size,
                            sequence_length=sequence_length)

# %%

x_batch, y_batch = next(generator)

# %%

print(x_batch.shape)
print(y_batch.shape)

# %%

batch = 0  # First sequence in the batch.
signal = 0  # First signal from the 20 input-signals.
seq = x_batch[batch, :, signal]
plt.plot(seq)

# %%

seq = y_batch[batch, :, signal]
plt.plot(seq)

# %%

validation_data = (np.expand_dims(x_test_scaled, axis=0),
                   np.expand_dims(y_test_scaled, axis=0))

# %%

model = Sequential()

# %%

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.GRU(300, return_sequences=True, input_shape=(None, num_x_signals,)))
model.add(tf.keras.layers.ReLU())
model.add(tf.keras.layers.Dense(300))
model.add(tf.keras.layers.LeakyReLU())
model.add(tf.keras.layers.Dense(num_y_signals))

# %%

warmup_steps = 50


# %%

def loss_mse_warmup(y_true, y_pred):
    """
    Calculate the Mean Squared Error between y_true and y_pred,
    but ignore the beginning "warmup" part of the sequences.

    y_true is the desired output.
    y_pred is the model's output.
    """

    # The shape of both input tensors are:
    # [batch_size, sequence_length, num_y_signals].

    # Ignore the "warmup" parts of the sequences
    # by taking slices of the tensors.
    y_true_slice = y_true[:, warmup_steps:, :]
    y_pred_slice = y_pred[:, warmup_steps:, :]

    # These sliced tensors both have this shape:
    # [batch_size, sequence_length - warmup_steps, num_y_signals]

    # Calculat the Mean Squared Error and use it as loss.
    mse = mean(square(y_true_slice - y_pred_slice))

    return mse


# %%

optimizer = RMSprop(lr=1e-3)

# %%

model.compile(loss=loss_mse_warmup, optimizer=optimizer)

# %%

model.summary()

# %%

path_checkpoint = '23_checkpoint.keras'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      monitor='val_loss',
                                      verbose=1,
                                      save_weights_only=True,
                                      save_best_only=True)

# %%

callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=5, verbose=1)

callback_tensorboard = TensorBoard(log_dir='./23_logs/',
                                   histogram_freq=0,
                                   write_graph=False)

callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.1,
                                       min_lr=1e-4,
                                       patience=0,
                                       verbose=1)
callbacks = [callback_early_stopping,
             callback_checkpoint,
             callback_tensorboard,
             callback_reduce_lr]

# %%

model.fit(x=generator,
          epochs=20,
          steps_per_epoch=100,
          validation_data=validation_data,
          callbacks=callbacks)

# %%

try:
    model.load_weights(path_checkpoint)
except Exception as error:
    print("Error trying to load checkpoint.")
    print(error)

result = model.evaluate(x=np.expand_dims(x_test_scaled, axis=0),
                        y=np.expand_dims(y_test_scaled, axis=0))

print("loss (test-set):", result)

# %%

y_train

# %%

# If you have several metrics you can use this instead.
if False:
    for res, metric in zip(result, model.metrics_names):
        print("{0}: {1:.3e}".format(metric, res))


def plot_comparison(train=True, name="out"):
    """
    Plot the predicted and true output-signals.

    :param name:
    :param start_idx: Start-index for the time-series.
    :param length: Sequence-length to process and plot.
    :param train: Boolean whether to use training- or test-set.
    """

    if train:
        # Use training-data.
        x = x_train_scaled
        y_true = y_train
    else:
        # Use test-data.
        x = x_test_scaled
        y_true = y_test

    # Input-signals for the model.
    x = np.expand_dims(x, axis=0)

    # Use the model to predict the output-signals.
    y_pred = model.predict(x)
    y_pred_rescaled = y_scaler.inverse_transform(y_pred[0])
    signal_pred = y_pred_rescaled
    signal_true = y_true

    plt.figure(figsize=(15, 5))
    plt.plot(signal_true, label='true')
    plt.plot(signal_pred, label='pred')

    # Plot grey box for warmup-period.
    p = plt.axvspan(0, warmup_steps, facecolor='black', alpha=0.15)

    # Plot labels etc.
    plt.ylabel("PV output")
    plt.legend()
    plt.show()
    plt.savefig(name + ".png")


# %%

x = ult_x_scaled
y_true = ult_y_scaled
x = np.expand_dims(x, axis=0)

# Use the model to predict the output-signals.
y_pred = model.predict(x)
y_pred_rescaled = y_scaler.inverse_transform(y_pred[0])
signal_pred = y_pred_rescaled
signal_true = y_true
plt.figure(figsize=(15, 5))
plt.plot(signal_true, label='true')
plt.plot(signal_pred, label='pred')
plt.savefig("final.png")

