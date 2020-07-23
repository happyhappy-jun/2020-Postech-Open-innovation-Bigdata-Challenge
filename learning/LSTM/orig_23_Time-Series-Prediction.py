#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler


# We need to import several things from Keras.

# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.backend import square, mean


# This was developed using Python 3.6 (Anaconda) and package versions:

# In[ ]:


tf.__version__


# In[ ]:


tf.keras.__version__


# In[ ]:


pd.__version__

import weather


# Download the data-set if you don't have it already. It is about 35 MB.

# In[ ]:


weather.maybe_download_and_extract()


# List of the cities used in the data-set.

# In[ ]:


cities = weather.cities
cities


# Load and resample the data so it has observations at regular time-intervals for every 60 minutes. Missing data-points are linearly interpolated. This takes about 30 seconds to run the first time but uses a cache-file so it loads very quickly the next time.

# In[ ]:


df = weather.load_resampled_data()


# These are the top rows of the data-set.

# In[ ]:


df.head()


# ### Missing Data
# 
# The two cities Esbjerg and Roskilde have missing data for the atmospheric pressure, as can be seen in the following two plots. 
# 
# Because we are using resampled data, we have filled in the missing values with new values that are linearly interpolated from the neighbouring values, which appears as long straight lines in these plots.
# 
# This may confuse the neural network. For simplicity, we will simply remove these two signals from the data.
# 
# But it is only short periods of data that are missing, so you could actually generate this data by creating a predictive model that generates the missing data from all the other input signals. Then you could add these generated values back into the data-set to fill the gaps.

# In[ ]:


df['Esbjerg']['Pressure'].plot();


# In[ ]:


df['Roskilde']['Pressure'].plot();


# Before removing these two signals, there are 20 input-signals in the data-set.

# In[ ]:


df.values.shape


# Then we remove the two signals that have missing data.

# In[ ]:


df.drop(('Esbjerg', 'Pressure'), axis=1, inplace=True)
df.drop(('Roskilde', 'Pressure'), axis=1, inplace=True)


# Now there are only 18 input-signals in the data.

# In[ ]:


df.values.shape


# We can verify that these two data-columns have indeed been removed.

# In[ ]:


df.head(1)


# ### Data Errors
# 
# There are some errors in this data. As shown in the plot below, the temperature in the city of Odense suddenly jumped to almost 50 degrees C. But the highest temperature ever measured in Denmark was only 36.4 degrees Celcius and the lowest was -31.2 C. So this is clearly a data error. However, we will not correct any data-errors in this tutorial.

# In[ ]:


df['Odense']['Temp']['2006-05':'2006-07'].plot();


# This can also be confirmed to be an error by considering the temperatures in some of the other cities in Denmark for that period, which was only around 10 degrees. Because the country is so small, it is not possible for one city in Denmark to have 50 degrees while another city only has 10 degrees.

# In[ ]:


df['Aarhus']['Temp']['2006-05':'2006-07'].plot();


# In[ ]:


df['Roskilde']['Temp']['2006-05':'2006-07'].plot();


# ### Add Data
# 
# We can add some input-signals to the data that may help our model in making predictions.
# 
# For example, given just a temperature of 10 degrees Celcius the model wouldn't know whether that temperature was measured during the day or the night, or during summer or winter. The model would have to infer this from the surrounding data-points which might not be very accurate for determining whether it's an abnormally warm winter, or an abnormally cold summer, or whether it's day or night. So having this information could make a big difference in how accurately the model can predict the next output.
# 
# Although the data-set does contain the date and time information for each observation, it is only used in the index so as to order the data. We will therefore add separate input-signals to the data-set for the day-of-year (between 1 and 366) and the hour-of-day (between 0 and 23).

# In[ ]:


df['Various', 'Day'] = df.index.dayofyear
df['Various', 'Hour'] = df.index.hour


# ### Target Data for Prediction
# 
# We will try and predict the future weather-data for this city.

# In[ ]:


target_city = 'Odense'


# We will try and predict these signals.

# In[ ]:


target_names = ['Temp', 'WindSpeed', 'Pressure']


# The following is the number of time-steps that we will shift the target-data. Our data-set is resampled to have an observation for each hour, so there are 24 observations for 24 hours.
# 
# If we want to predict the weather 24 hours into the future, we shift the data 24 time-steps. If we want to predict the weather 7 days into the future, we shift the data 7 * 24 time-steps.

# In[ ]:


shift_days = 1
shift_steps = shift_days * 24  # Number of hours.


# Create a new data-frame with the time-shifted data.
# 
# **Note the negative time-shift!**

# In[ ]:


df_targets = df[target_city][target_names].shift(-shift_steps)


# **WARNING!** You should double-check that you have shifted the data in the right direction! We want to predict the future, not the past!
# 
# The shifted data-frame is confusing because Pandas keeps the original time-stamps even though we have shifted the data. You can check the time-shift is correct by comparing the original and time-shifted data-frames.
# 
# This is the first `shift_steps + 5` rows of the original data-frame:

# In[ ]:


df[target_city][target_names].head(shift_steps + 5)


# The following is the first 5 rows of the time-shifted data-frame. This should be identical to the last 5 rows shown above from the original data, except for the time-stamp.

# In[ ]:


df_targets.head(5)


# The time-shifted data-frame has the same length as the original data-frame, but the last observations are `NaN` (not a number) because the data has been shifted backwards so we are trying to shift data that does not exist in the original data-frame.

# In[ ]:


df_targets.tail()


# ### NumPy Arrays
# 
# We now convert the Pandas data-frames to NumPy arrays that can be input to the neural network. We also remove the last part of the numpy arrays, because the target-data has `NaN` for the shifted period, and we only want to have valid data and we need the same array-shapes for the input- and output-data.
# 
# These are the input-signals:

# In[ ]:


x_data = df.values[0:-shift_steps]


# In[ ]:


print(type(x_data))
print("Shape:", x_data.shape)


# These are the output-signals (or target-signals):

# In[ ]:


y_data = df_targets.values[:-shift_steps]


# In[ ]:


print(type(y_data))
print("Shape:", y_data.shape)


# This is the number of observations (aka. data-points or samples) in the data-set:

# In[ ]:


num_data = len(x_data)
num_data


# This is the fraction of the data-set that will be used for the training-set:

# In[ ]:


train_split = 0.9


# This is the number of observations in the training-set:

# In[ ]:


num_train = int(train_split * num_data)
num_train


# This is the number of observations in the test-set:

# In[ ]:


num_test = num_data - num_train
num_test


# These are the input-signals for the training- and test-sets:

# In[ ]:


x_train = x_data[0:num_train]
x_test = x_data[num_train:]
len(x_train) + len(x_test)


# These are the output-signals for the training- and test-sets:

# In[ ]:


y_train = y_data[0:num_train]
y_test = y_data[num_train:]
len(y_train) + len(y_test)


# This is the number of input-signals:

# In[ ]:


num_x_signals = x_data.shape[1]
num_x_signals


# This is the number of output-signals:

# In[ ]:


num_y_signals = y_data.shape[1]
num_y_signals


# ### Scaled Data
# 
# The data-set contains a wide range of values:

# In[ ]:


print("Min:", np.min(x_train))
print("Max:", np.max(x_train))


# The neural network works best on values roughly between -1 and 1, so we need to scale the data before it is being input to the neural network. We can use `scikit-learn` for this.
# 
# We first create a scaler-object for the input-signals.

# In[ ]:


x_scaler = MinMaxScaler()


# We then detect the range of values from the training-data and scale the training-data.

# In[ ]:


x_train_scaled = x_scaler.fit_transform(x_train)


# Apart from a small rounding-error, the data has been scaled to be between 0 and 1.

# In[ ]:


print("Min:", np.min(x_train_scaled))
print("Max:", np.max(x_train_scaled))


# We use the same scaler-object for the input-signals in the test-set.

# In[ ]:


x_test_scaled = x_scaler.transform(x_test)


# The target-data comes from the same data-set as the input-signals, because it is the weather-data for one of the cities that is merely time-shifted. But the target-data could be from a different source with different value-ranges, so we create a separate scaler-object for the target-data.

# In[ ]:


y_scaler = MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)


# ## Data Generator
# 
# The data-set has now been prepared as 2-dimensional numpy arrays. The training-data has almost 300k observations, consisting of 20 input-signals and 3 output-signals.
# 
# These are the array-shapes of the input and output data:

# In[ ]:


print(x_train_scaled.shape)
print(y_train_scaled.shape)


# Instead of training the Recurrent Neural Network on the complete sequences of almost 300k observations, we will use the following function to create a batch of shorter sub-sequences picked at random from the training-data.

# In[ ]:


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


# We will use a large batch-size so as to keep the GPU near 100% work-load. You may have to adjust this number depending on your GPU, its RAM and your choice of `sequence_length` below.

# In[ ]:


batch_size = 256


# We will use a sequence-length of 1344, which means that each random sequence contains observations for 8 weeks. One time-step corresponds to one hour, so 24 x 7 time-steps corresponds to a week, and 24 x 7 x 8 corresponds to 8 weeks.

# In[ ]:


sequence_length = 24 * 7 * 8
sequence_length


# We then create the batch-generator.

# In[ ]:


generator = batch_generator(batch_size=batch_size,
                            sequence_length=sequence_length)


# We can then test the batch-generator to see if it works.

# In[ ]:


x_batch, y_batch = next(generator)


# This gives us a random batch of 256 sequences, each sequence having 1344 observations, and each observation having 20 input-signals and 3 output-signals.

# In[ ]:


print(x_batch.shape)
print(y_batch.shape)


# We can plot one of the 20 input-signals as an example.

# In[ ]:


batch = 0   # First sequence in the batch.
signal = 0  # First signal from the 20 input-signals.
seq = x_batch[batch, :, signal]
plt.plot(seq)


# We can also plot one of the output-signals that we want the model to learn how to predict given all those 20 input signals.

# In[ ]:


seq = y_batch[batch, :, signal]
plt.plot(seq)


# ### Validation Set
# 
# The neural network trains quickly so we can easily run many training epochs. But then there is a risk of overfitting the model to the training-set so it does not generalize well to unseen data. We will therefore monitor the model's performance on the test-set after each epoch and only save the model's weights if the performance is improved on the test-set.
# 
# The batch-generator randomly selects a batch of short sequences from the training-data and uses that during training. But for the validation-data we will instead run through the entire sequence from the test-set and measure the prediction accuracy on that entire sequence.

# In[ ]:


validation_data = (np.expand_dims(x_test_scaled, axis=0),
                   np.expand_dims(y_test_scaled, axis=0))


# ## Create the Recurrent Neural Network
# 
# We are now ready to create the Recurrent Neural Network (RNN). We will use the Keras API for this because of its simplicity. See Tutorial #03-C for a tutorial on Keras and Tutorial #20 for more information on Recurrent Neural Networks.

# In[ ]:


model = Sequential()


# We can now add a Gated Recurrent Unit (GRU) to the network. This will have 512 outputs for each time-step in the sequence.
# 
# Note that because this is the first layer in the model, Keras needs to know the shape of its input, which is a batch of sequences of arbitrary length (indicated by `None`), where each observation has a number of input-signals (`num_x_signals`).

# In[ ]:


model.add(GRU(units=512,
              return_sequences=True,
              input_shape=(None, num_x_signals,)))


# The GRU outputs a batch of sequences of 512 values. We want to predict 3 output-signals, so we add a fully-connected (or dense) layer which maps 512 values down to only 3 values.
# 
# The output-signals in the data-set have been limited to be between 0 and 1 using a scaler-object. So we also limit the output of the neural network using the Sigmoid activation function, which squashes the output to be between 0 and 1.

# In[ ]:


model.add(Dense(num_y_signals, activation='sigmoid'))


# A problem with using the Sigmoid activation function, is that we can now only output values in the same range as the training-data.
# 
# For example, if the training-data only has temperatures between -20 and +30 degrees, then the scaler-object will map -20 to 0 and +30 to 1. So if we limit the output of the neural network to be between 0 and 1 using the Sigmoid function, this can only be mapped back to temperature values between -20 and +30.
# 
# We can use a linear activation function on the output instead. This allows for the output to take on arbitrary values. It might work with the standard initialization for a simple network architecture, but for more complicated network architectures e.g. with more layers, it might be necessary to initialize the weights with smaller values to avoid `NaN` values during training. You may need to experiment with this to get it working.

# In[ ]:


if False:
    from tensorflow.python.keras.initializers import RandomUniform

    # Maybe use lower init-ranges.
    init = RandomUniform(minval=-0.05, maxval=0.05)

    model.add(Dense(num_y_signals,
                    activation='linear',
                    kernel_initializer=init))


# ### Loss Function
# 
# We will use Mean Squared Error (MSE) as the loss-function that will be minimized. This measures how closely the model's output matches the true output signals.
# 
# However, at the beginning of a sequence, the model has only seen input-signals for a few time-steps, so its generated output may be very inaccurate. Using the loss-value for the early time-steps may cause the model to distort its later output. We therefore give the model a "warmup-period" of 50 time-steps where we don't use its accuracy in the loss-function, in hope of improving the accuracy for later time-steps.

# In[ ]:


warmup_steps = 50


# In[ ]:


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


# ### Compile Model
# 
# This is the optimizer and the beginning learning-rate that we will use.

# In[ ]:


optimizer = RMSprop(lr=1e-3)


# We then compile the Keras model so it is ready for training.

# In[ ]:


model.compile(loss=loss_mse_warmup, optimizer=optimizer)


# This is a very small model with only two layers. The output shape of `(None, None, 3)` means that the model will output a batch with an arbitrary number of sequences, each of which has an arbitrary number of observations, and each observation has 3 signals. This corresponds to the 3 target signals we want to predict.

# In[ ]:


model.summary()


# ### Callback Functions
# 
# During training we want to save checkpoints and log the progress to TensorBoard so we create the appropriate callbacks for Keras.
# 
# This is the callback for writing checkpoints during training.

# In[ ]:


path_checkpoint = '23_checkpoint.keras'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      monitor='val_loss',
                                      verbose=1,
                                      save_weights_only=True,
                                      save_best_only=True)


# This is the callback for stopping the optimization when performance worsens on the validation-set.

# In[ ]:


callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=5, verbose=1)


# This is the callback for writing the TensorBoard log during training.

# In[ ]:


callback_tensorboard = TensorBoard(log_dir='./23_logs/',
                                   histogram_freq=0,
                                   write_graph=False)


# This callback reduces the learning-rate for the optimizer if the validation-loss has not improved since the last epoch (as indicated by `patience=0`). The learning-rate will be reduced by multiplying it with the given factor. We set a start learning-rate of 1e-3 above, so multiplying it by 0.1 gives a learning-rate of 1e-4. We don't want the learning-rate to go any lower than this.

# In[ ]:


callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.1,
                                       min_lr=1e-4,
                                       patience=0,
                                       verbose=1)


# In[ ]:


callbacks = [callback_early_stopping,
             callback_checkpoint,
             callback_tensorboard,
             callback_reduce_lr]


# ## Train the Recurrent Neural Network
# 
# We can now train the neural network.
# 
# Note that a single "epoch" does not correspond to a single processing of the training-set, because of how the batch-generator randomly selects sub-sequences from the training-set. Instead we have selected `steps_per_epoch` so that one "epoch" is processed in a few minutes.
# 
# With these settings, each "epoch" took about 2.5 minutes to process on a GTX 1070. After 14 "epochs" the optimization was stopped because the validation-loss had not decreased for 5 "epochs". This optimization took about 35 minutes to finish.
# 
# Also note that the loss sometimes becomes `NaN` (not-a-number). This is often resolved by restarting and running the Notebook again. But it may also be caused by your neural network architecture, learning-rate, batch-size, sequence-length, etc. in which case you may have to modify those settings.

# In[ ]:


model.fit(x=generator,          epochs=3,          steps_per_epoch=100,          validation_data=validation_data,          callbacks=callbacks)


# ### Load Checkpoint
# 
# Because we use early-stopping when training the model, it is possible that the model's performance has worsened on the test-set for several epochs before training was stopped. We therefore reload the last saved checkpoint, which should have the best performance on the test-set.

# In[ ]:


try:
    model.load_weights(path_checkpoint)
except Exception as error:
    print("Error trying to load checkpoint.")
    print(error)


# ## Performance on Test-Set
# 
# We can now evaluate the model's performance on the test-set. This function expects a batch of data, but we will just use one long time-series for the test-set, so we just expand the array-dimensionality to create a batch with that one sequence.

# In[ ]:


result = model.evaluate(x=np.expand_dims(x_test_scaled, axis=0),
                        y=np.expand_dims(y_test_scaled, axis=0))


# In[ ]:


print("loss (test-set):", result)


# In[ ]:


# If you have several metrics you can use this instead.
if False:
    for res, metric in zip(result, model.metrics_names):
        print("{0}: {1:.3e}".format(metric, res))


# ## Generate Predictions
# 
# This helper-function plots the predicted and true output-signals.

# In[ ]:


def plot_comparison(start_idx, length=100, train=True):
    """
    Plot the predicted and true output-signals.
    
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
    
    # End-index for the sequences.
    end_idx = start_idx + length
    
    # Select the sequences from the given start-index and
    # of the given length.
    x = x[start_idx:end_idx]
    y_true = y_true[start_idx:end_idx]
    
    # Input-signals for the model.
    x = np.expand_dims(x, axis=0)
    print("x dim: {}".format(x.shape))
    # Use the model to predict the output-signals.
    y_pred = model.predict(x)
    
    # The output of the model is between 0 and 1.
    # Do an inverse map to get it back to the scale
    # of the original data-set.
    y_pred_rescaled = y_scaler.inverse_transform(y_pred[0])
    
    # For each output-signal.
    for signal in range(len(target_names)):
        # Get the output-signal predicted by the model.
        signal_pred = y_pred_rescaled[:, signal]
        
        # Get the true output-signal from the data-set.
        signal_true = y_true[:, signal]

        # Make the plotting-canvas bigger.
        plt.figure(figsize=(15,5))
        
        # Plot and compare the two signals.
        plt.plot(signal_true, label='true')
        plt.plot(signal_pred, label='pred')
        
        # Plot grey box for warmup-period.
        p = plt.axvspan(0, warmup_steps, facecolor='black', alpha=0.15)
        
        # Plot labels etc.
        plt.ylabel(target_names[signal])
        plt.legend()
        plt.show()


# We can now plot an example of predicted output-signals. It is important to understand what these plots show, as they are actually a bit more complicated than you might think.
# 
# These plots only show the output-signals and not the 20 input-signals used to predict the output-signals. The time-shift between the input-signals and the output-signals is held fixed in these plots. The model **always** predicts the output-signals e.g. 24 hours into the future (as defined in the `shift_steps` variable above). So the plot's x-axis merely shows how many time-steps of the input-signals have been seen by the predictive model so far.
# 
# The prediction is not very accurate for the first 30-50 time-steps because the model has seen very little input-data at this point.
# The model generates a single time-step of output data for each time-step of the input-data, so when the model has only run for a few time-steps, it knows very little of the history of the input-signals and cannot make an accurate prediction. The model needs to "warm up" by processing perhaps 30-50 time-steps before its predicted output-signals can be used.
# 
# That is why we ignore this "warmup-period" of 50 time-steps when calculating the mean-squared-error in the loss-function. The "warmup-period" is shown as a grey box in these plots.
# 
# Let us start with an example from the training-data. This is data that the model has seen during training so it should perform reasonably well on this data.

# In[ ]:


plot_comparison(start_idx=100000, length=1000, train=True)


# The model was able to predict the overall oscillations of the temperature quite well but the peaks were sometimes inaccurate. For the wind-speed, the overall oscillations are predicted reasonably well but the peaks are quite inaccurate. For the atmospheric pressure, the overall curve-shape has been predicted although there seems to be a slight lag and the predicted curve has a lot of noise compared to the smoothness of the original signal.

# ### Strange Example
# 
# The following is another example from the training-set.
# 
# Note how the temperature does not oscillate very much within each day (this plot shows almost 42 days). The temperature normally oscillates within each day, see e.g. the plot above where the daily temperature-oscillation is very clear. It is unclear whether this period had unusually stable temperature, or if perhaps there's a data-error.

# In[ ]:


plot_comparison(start_idx=200000, length=1000, train=True)


# As a check, we can plot this signal directly from the resampled data-set, which looks similar.

# In[ ]:


df['Odense']['Temp'][200000:200000+1000].plot();


# We can plot the same period from the original data that has not been resampled. It also looks similar.
# 
# So either the temperature was unusually stable for a part of this period, or there is a data-error in the raw data that was obtained from the internet weather-database.

# In[ ]:


df_org = weather.load_original_data()
df_org.xs('Odense')['Temp']['2002-12-23':'2003-02-04'].plot();


# ### Example from Test-Set
# 
# Now consider an example from the test-set. The model has not seen this data during training.
# 
# The temperature is predicted reasonably well, although the peaks are sometimes inaccurate.
# 
# The wind-speed has not been predicted so well. The daily oscillation-frequency seems to match, but the center-level and the peaks are quite inaccurate. A guess would be that the wind-speed is difficult to predict from the given input data, so the model has merely learnt to output sinusoidal oscillations in the daily frequency and approximately at the right center-level.
# 
# The atmospheric pressure is predicted reasonably well, except for a lag and a more noisy signal than the true time-series.

# In[ ]:




