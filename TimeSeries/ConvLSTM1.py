# -*- coding: utf-8 -*-
"""
Created on Sun May 10 23:04:16 2020

@author: Mitchell
"""

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
### Imports
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import ResidualAnalysis, ValidationSplit
from lstm_utils import load_training, ConvLSTM, DataSequence
import tensorflow as tf
import time as tm
import os


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
### Load Data
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Read Test Stock Data from CSV
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
filename = "../Data/VTI-TimeSeriesDaily-2020-05-03.txt"
VTI = pd.read_csv(filename, index_col = 'date', encoding = "UTF-8")
VTI.index = VTI.index.astype('datetime64[ns]')

# Visualize Data
#~~~~~~~~~~~~~~~
plt.figure()
VTI['4. close'].plot(label = 'Close', linewidth = 1)
VTI['12. close adj_1'].plot(label = 'Adj. for Splits', linewidth = 1)
VTI['18. close adj_2'].plot(label = 'Adj. for Splits & Div.', linewidth = 1)
plt.legend()
plt.title('VTI Daily Closing Price')
plt.xlabel('Date')
plt.ylabel('Price Per Share ($)')
plt.show()

# Grab Data from Dataframe
#~~~~~~~~~~~~~~~~~~~~~~~~~
close = np.flip(VTI['5. adjusted close'].to_numpy())
datetime = np.flip(VTI.index.to_numpy())

# Split Data into Training and Validation
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
validation_ratio = .2
training_close , validation_close = ValidationSplit(close, validation_ratio)
training_datetime , validation_datetime = ValidationSplit(datetime, validation_ratio)
validation_size = len(validation_close)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
### Full LSTM Model
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
# Model Parameters
#~~~~~~~~~~~~~~~~~
tf.keras.backend.set_floatx('float64')
input_dim = 1
batch_size = 100
lstm_units = 64
features = 32
kernel_size = 3
stride = 1
temporal = False
max_pool = False
dense_units = 64
max_norm = None,
dropout = 0

train_full_model = True
if train_full_model:
    print('Training Full LSTM Model\n'+'-'*24)
    
    # Load Training Data
    #~~~~~~~~~~~~~~~~~~~
    print('Loading Training and Validation Data for Full Model')
    data_folder = "../Data/S&P500/"
    training_dataset, validation_dataset = load_training(data_folder,
                                                         validation_ratio = .1)
    
    # Build Model
    #~~~~~~~~~~~~
    full_model = ConvLSTM(input_dim,
                          batch_size,
                          lstm_units,
                          features = features,
                          kernel_size = kernel_size,
                          stride = stride,
                          temporal = temporal,
                          max_pool = max_pool,
                          dense_units = dense_units,
                          max_norm = max_norm,
                          dropout = dropout)
    full_model.build(tf.TensorShape([batch_size, None, input_dim]))
    full_model.summary()
    
    # Training Parameters
    #~~~~~~~~~~~~~~~~~~~~
    seq_length = 100
    steps_per_epoch = 10
    validation_steps = 1
    epochs = 10
    
    # Adjust Datasets
    #~~~~~~~~~~~~~~~~
    # That is remove all sample sequences shorter than seq_length+1
    indxs = []
    for i in range(len(training_dataset)):
        if training_dataset[i].shape[0] >= seq_length+1:
            indxs.append(i)
    training_dataset = training_dataset[indxs]
    indxs = []
    for i in range(len(validation_dataset)):
        if validation_dataset[i].shape[0] >= seq_length+1:
            indxs.append(i)
    validation_dataset = validation_dataset[indxs]
    
    # Cost Function
    #~~~~~~~~~~~~~~
    cost_function = tf.keras.losses.MAE
    
    # Learning Rate Schedule
    #~~~~~~~~~~~~~~~~~~~~~~~
    lr_0 = .01
    decay_rate = .98
    lr_decay = lambda t: lr_0 * decay_rate**t
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lr_decay)
    
    # Optimizer
    #~~~~~~~~~~
    optimizer = tf.keras.optimizers.Adam()
    
    # Callbacks
    #~~~~~~~~~~
    callbacks = [lr_schedule]
    
    # Generators for Batches
    #~~~~~~~~~~~~~~~~~~~~~~~
    training_seq = DataSequence(training_dataset, batch_size, seq_length,
                                steps_per_epoch, weighted = True,
                                  return_sequences = False)
    validation_seq = DataSequence(validation_dataset, batch_size, seq_length,
                                  validation_steps, weighted = True,
                                  return_sequences = False)
    
    # Compile Model
    #~~~~~~~~~~~~~~
    full_model.compile(optimizer = optimizer,
                       loss = cost_function)
    
    # Train model
    #~~~~~~~~~~~~
    tic = tm.perf_counter()
    history = full_model.fit(training_seq,
                             epochs = epochs,
                             callbacks = callbacks,
                             validation_data = validation_seq,
                             use_multiprocessing = True)
    toc = tm.perf_counter()
    print(f"Trained Model in {(toc - tic)/60:0.1f} minutes")

    # Save Model
    #~~~~~~~~~~~
    save_weights = True
    if save_weights:
        checkpoint_dir = '.\\training_checkpoints\\full_model_1'
        checkpoint_prefix = os.path.join(checkpoint_dir, "model_ckpt")
        full_model.save_weights(checkpoint_prefix)
        print('Full Model weights saved to files: '+checkpoint_prefix+'.*')
    
# ReBuild Model With Batch-size = 1
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
checkpoint_dir = '.\\training_checkpoints\\full_model_1'
checkpoint_prefix = os.path.join(checkpoint_dir, "model_ckpt")
full_model = ConvLSTM(input_dim,
                      1,
                      lstm_units,
                      features = features,
                      kernel_size = kernel_size,
                      stride = stride,
                      temporal = temporal,
                      max_pool = max_pool,
                      dense_units = dense_units,
                      max_norm = max_norm,
                      dropout = dropout)
full_model.load_weights(checkpoint_prefix)
full_model.build(tf.TensorShape([1, None, input_dim]))
    

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
### Full Model Residuals
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Reshape Data For Input
#~~~~~~~~~~~~~~~~~~~~~~~
close_input = close[:-1].reshape(1, close.shape[0]-1, 1)

# Get Residuals
#~~~~~~~~~~~~~~
residuals = close[1:] - full_model(close_input).numpy().reshape(close_input.shape[1])

# Residual Anlysis
#~~~~~~~~~~~~~~~~~
ResidualAnalysis(datetime[1:], residuals, nlags = 252)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
### Test LSTM Model
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

train_test_model = True
if train_test_model:
    print('Training Test LSTM Model\n'+'-'*24)
    
    # Load Training Data
    #~~~~~~~~~~~~~~~~~~~
    print('Loading Training and Validation Data for Test Model')
    data_folder = "../Data/S&P500/"
    training_dataset, validation_dataset = load_training(data_folder,
                                                         validation_ratio = .1,
                                                         max_date = '2016-07-20')
    
    # Build Model
    #~~~~~~~~~~~~
    test_model = ConvLSTM(input_dim,
                          batch_size,
                          lstm_units,
                          features = features,
                          kernel_size = kernel_size,
                          stride = stride,
                          temporal = temporal,
                          max_pool = max_pool,
                          dense_units = dense_units,
                          max_norm = max_norm,
                          dropout = dropout)
    test_model.build(tf.TensorShape([batch_size, None, input_dim]))
    test_model.summary()
    
    # Training Parameters
    #~~~~~~~~~~~~~~~~~~~~
    seq_length = 100
    steps_per_epoch = 10
    validation_steps = 1
    epochs = 10
    
    # Adjust Datasets
    #~~~~~~~~~~~~~~~~
    # That is remove all sample sequences shorter than seq_length+1
    indxs = []
    for i in range(len(training_dataset)):
        if training_dataset[i].shape[0] >= seq_length+1:
            indxs.append(i)
    training_dataset = training_dataset[indxs]
    indxs = []
    for i in range(len(validation_dataset)):
        if validation_dataset[i].shape[0] >= seq_length+1:
            indxs.append(i)
    validation_dataset = validation_dataset[indxs]
    
    # Cost Function
    #~~~~~~~~~~~~~~
    cost_function = tf.keras.losses.MAE
    
    # Learning Rate Schedule
    #~~~~~~~~~~~~~~~~~~~~~~~
    lr_0 = .01
    decay_rate = .98
    lr_decay = lambda t: lr_0 * decay_rate**t
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lr_decay)
    
    # Optimizer
    #~~~~~~~~~~
    optimizer = tf.keras.optimizers.Adam()
    
    # Callbacks
    #~~~~~~~~~~
    callbacks = [lr_schedule]
    
    # Generators for Batches
    #~~~~~~~~~~~~~~~~~~~~~~~
    training_seq = DataSequence(training_dataset, batch_size, seq_length,
                                steps_per_epoch, weighted = True)
    validation_seq = DataSequence(validation_dataset, batch_size, seq_length,
                                  validation_steps, weighted = True)
    
    # Compile Model
    #~~~~~~~~~~~~~~
    test_model.compile(optimizer = optimizer,
                       loss = cost_function)
    
    # Train model
    #~~~~~~~~~~~~
    tic = tm.perf_counter()
    history = test_model.fit(training_seq,
                             epochs = epochs,
                             callbacks = callbacks,
                             validation_data = validation_seq,
                             use_multiprocessing = True)
    toc = tm.perf_counter()
    print(f"Trained Model in {(toc - tic)/60:0.1f} minutes")

    # Save Model
    #~~~~~~~~~~~
    save_weights = True
    if save_weights:
        checkpoint_dir = '.\\training_checkpoints\\test_model_1'
        checkpoint_prefix = os.path.join(checkpoint_dir, "model_ckpt")
        test_model.save_weights(checkpoint_prefix)
        print('Test Model weights saved to files: '+checkpoint_prefix+'.*')
    
# ReBuild Model With Batch-size = 1
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
checkpoint_dir = '.\\training_checkpoints\\test_model_1'
checkpoint_prefix = os.path.join(checkpoint_dir, "model_ckpt")
test_model = ConvLSTM(input_dim,
                      1,
                      lstm_units,
                      features = features,
                      kernel_size = kernel_size,
                      stride = stride,
                      temporal = temporal,
                      max_pool = max_pool,
                      dense_units = dense_units,
                      max_norm = max_norm,
                      dropout = dropout)
test_model.load_weights(checkpoint_prefix)
test_model.build(tf.TensorShape([1, None, input_dim]))
    

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
### Test Model Validation
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('Running Test Model Validation'+'\n'+'-'*29)

# Reshape Data For Input
#~~~~~~~~~~~~~~~~~~~~~~~
training_close_input = training_close[:-1].reshape(1, training_close.shape[0]-1, 1)

# Get Model Predictions
#~~~~~~~~~~~~~~~~~~~~~~
pred_close = test_model.get_prediction(training_close_input, steps = validation_size)

# Get Erros
#~~~~~~~~~~
abs_error = np.abs(validation_close - pred_close)
ame , ame_std = abs_error.mean() , abs_error.std()

# Plot Predictions
#~~~~~~~~~~~~~~~~~
plt.figure()
plt.plot(training_datetime[-validation_size:], training_close[-validation_size:],
         'b', linewidth = 1, label = 'Training')
plt.plot(validation_datetime, validation_close, 'k', linewidth = 1,
         label = 'Validation')
plt.plot(validation_datetime, pred_close, 'r', linewidth = 1,
         label = 'Prediction')
plt.legend()
plt.title('Naive Drift Model Prediction Test')
plt.xlabel('Date')
plt.ylabel('VTI Closing Price Per Share ($)')
plt.show()

# Print Statistics
#~~~~~~~~~~~~~~~~~
print('Mean and Std. of validation absolute error: mu = {:6.4f}, sigma = {:6.4f}'\
      .format(ame, ame_std))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#----------------------------------END FILE------------------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~