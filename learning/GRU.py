import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from keras.models import Sequential
from keras.layers import Input, Dense, GRU, Embedding
from keras.optimizers import RMSprop, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras import backend as K





#%%
from sklearn.preprocessing import MinMaxScaler
df = pd.read_csv("datefrom1st.csv")
df.index = df.datetime
df = df.drop(["Unnamed: 0",'datetime', 'percipitation', 'air_pressure', 'sea_level_pressure', 'wind_degree'], axis = 1)

#%%

def multivariate_data(ds, target, start_index, end_index, history_size, target_size, step, single_step=False):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(ds) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i, step)
    data.append(ds[indices])

    if single_step:
      labels.append(target[i+target_size])
    else:
      labels.append(target[i:i+target_size])

  return np.array(data), np.array(labels)

#%%

BUFFER_SIZE = 10000
BATCH_SIZE = 32
TEST_RATIO = 0.3
TRAIN_SIZE = int((1-TEST_RATIO)*df.shape[0])

#%%

df.head(5)


#%%

shift_days = 1
shift_steps = shift_days * 24 * 4
df_targets = df.result.shift(-shift_steps)

#%%

# X_train, X_test, y_train, y_test = train_test_split(dataset[:,1:], dataset[:,0], test_size=0.20, shuffle=False)
# 

#%%

df.head(15)

#%%

x_data = df.values[0:-shift_steps]
print(type(x_data))
print("Shape:", x_data.shape)
#%%

y_data = df_targets.values[:-shift_steps]
print(type(y_data))
print("Shape:", y_data.shape)
    
#%%

num_data = len(x_data)
train_split = 0.9
num_train = int(train_split * num_data)
num_test = num_data - num_train
x_train = x_data[0:num_train]
x_test = x_data[num_train:]
y_train = y_data[0:num_train]
y_test = y_data[num_train:]
num_x_signals = x_data.shape[1]
num_y_signals = 1
#%%

x_scaler = MinMaxScaler()
x_train_scaled = x_scaler.fit_transform(x_train)
print("Min:", np.min(x_train_scaled))
print("Max:", np.max(x_train_scaled))

#%%

x_test_scaled = x_scaler.transform(x_test)
y_scaler = MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1))

#%%



#%%

print(x_train_scaled.shape)
print(y_train_scaled.shape)

#%%

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
        y_shape = (batch_size, sequence_length, num_y_signals)
        y_batch = np.zeros(shape=y_shape, dtype=np.float16)
        # Fill the batch with random sequences of data.
        for i in range(batch_size):
            # Get a random start-index.
            # This points somewhere into the training-data.
            idx = np.random.randint(num_train - sequence_length)
            
            # Copy the sequences of data starting at this index.
            x_batch[i] = x_train_scaled[idx:idx+sequence_length]
            y_batch[i] = y_train_scaled[idx:idx+sequence_length]

        yield (x_batch, y_batch)

#%%

batch_size = 32 
sequence_length = 24 * 7 * 4*3

#%%

generator = batch_generator(batch_size=batch_size,
 sequence_length=sequence_length)

#%%

x_batch, y_batch = next(generator)

#%%

print(x_batch.shape)
print(y_batch.shape)


validation_data = (np.expand_dims(x_test_scaled, axis=0),
 np.expand_dims(y_test_scaled, axis=0))

#%%
model = keras.models.Sequential()
model.add(keras.layers.GRU(units=512,
 return_sequences=True, input_shape = (None, num_x_signals)))
model.add(keras.layers.Dense(num_y_signals, activation='sigmoid'))



#%%

warmup_steps = 50

#%%

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
    # Calculate the MSE loss for each value in these tensors.
    # This outputs a 3-rank tensor of the same shape.
    loss = tf.keras.losses.MSE(y_true_slice,y_pred_slice)
    # Keras may reduce this across the first axis (the batch)
    # but the semantics are unclear, so to be sure we use
    # the loss across the entire tensor, we reduce it to a
    # single scalar with the mean function.
    loss_mean = tf.reduce_mean(loss)
    return loss_mean

optimizer = RMSprop()
model.compile(loss=loss_mse_warmup, optimizer=keras.optimizers.SGD(learning_rate=0.001), metrics=['accuracy'])

model.summary()
#%%
path_checkpoint = '23_checkpoint.keras'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
 monitor='val_loss',
verbose=1,
save_weights_only=True,
save_best_only=True)


#%%

callback_early_stopping = EarlyStopping(monitor='val_loss',patience=5, verbose=1)

#%%

callback_tensorboard = TensorBoard(log_dir='./23_logs/', histogram_freq=0, write_graph=False)
callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, min_lr=1e-4, patience=0,verbose=1)
callbacks = [callback_early_stopping, callback_checkpoint, callback_tensorboard, callback_reduce_lr]

#callbacks = [callback_early_stopping, callback_reduce_lr]

#%%



#%%

model.fit_generator(generator=generator,
 epochs=10000,verbose = 1, steps_per_epoch=100,
 validation_data=validation_data,
 callbacks=callbacks)

#%%

try:
    model.load_weights(path_checkpoint)
except Exception as error:
    print("Error trying to load checkpoint.")
    print(error)

#%%

result = model.evaluate(x=np.expand_dims(x_test_scaled, axis=0),
 y=np.expand_dims(y_test_scaled, axis=0))

#%%


print("loss (test-set):", result)

model.save('my_model.h5')
