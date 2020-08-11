# -*- coding: utf-8 -*-
"""
Created on Sat May  9 12:54:05 2020

@author: Mitchell
"""

# Imports
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from utils import BoxCox
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, BatchNormalization, Dropout, Dense,\
    Activation, TimeDistributed, Conv1D, MaxPool1D
from tensorflow.keras.utils import Sequence
import numpy as np
import pandas as pd
import glob, random


# Define LSTM Model Class
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class my_LSTM(Model):
    def __init__(self,
                 input_dim,
                 batch_size,
                 lstm_units,
                 dense_units = 0,
                 max_norm = None,
                 dropout = 0.):
        # Call to Super
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        super(my_LSTM, self).__init__()
        
         # Save Parameters
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.max_norm = max_norm
        self.dropout = dropout
        
         # Build Model
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Input Layer
        x = Input(shape=(None,input_dim), batch_size = batch_size, name = 'Input')
        
        # First LSTM Layer with BatchNormalization and Dropout
        y = LSTM(lstm_units,
                 return_sequences=True,
                 recurrent_initializer='glorot_uniform',
                 stateful=True)(x)
        y = BatchNormalization()(y)
        if dropout > 0:
            y = Dropout(dropout)(y)
        
        # Second LSTM Layer with BatchNormalization and Dropout
        y = LSTM(lstm_units,
                 return_sequences=True,
                 recurrent_initializer='glorot_uniform',
                 stateful=True)(y)
        y = BatchNormalization()(y)
        if dropout > 0:
            y = Dropout(dropout)(y)
        
        # Dense Layer with Batch Normalization and Dropout
        if max_norm:
            kernel_constraint = tf.keras.constraints.MaxNorm(max_value=max_norm)
        else:
            kernel_constraint = None
        if dense_units > 0:
            y = Dense(dense_units, kernel_constraint = kernel_constraint)(y)
            y = BatchNormalization()(y)
            y = Activation('relu')(y)
            if dropout > 0:
                y = Dropout(dropout)(y)
        
        # Output Layer
        y = Dense(1)(y)
        
        # Full Model
        self.model = Model(inputs = x, outputs = y)
        
    def call(self, x):
        return self.model(x)
    
    def get_prediction(self, x, steps = 1):
        
        # reset model states
        self.model.reset_states()
        
        # initiate array for predictions
        predictions = np.zeros(steps)
        
        # initiate model using x and grab first prediction
        pred = self.model(x)
        pred = pred[:,-1:,:]
        predictions[0] = pred.numpy().reshape(1)[0]
        
        # iteratively generate next steps
        for i in range(1, steps):
            pred = self.model(pred)
            predictions[i] = pred.numpy().reshape(1)[0]
        
        return predictions


# Define Conv LSTM Model Class
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class ConvLSTM(Model):
    def __init__(self,
                 input_dim,
                 batch_size,
                 lstm_units,
                 features = 32,
                 kernel_size = 3,
                 stride = 1,
                 temporal = False,
                 max_pool = False,
                 dense_units = 0,
                 max_norm = None,
                 dropout = 0.):
        # Call to Super
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        super(ConvLSTM, self).__init__()
        
         # Save Parameters
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.lstm_units = lstm_units
        self.features = features
        self.kernel_size = kernel_size
        self.stride = stride
        self.temporal = temporal
        self.max_pool = max_pool
        self.dense_units = dense_units
        self.max_norm = max_norm
        self.dropout = dropout
        
         # Build Model
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Input Layer
        x = Input(shape=(None,input_dim), batch_size = batch_size, name = 'Input')
        
        # 1-D Convolutional Layer with BatchNormalization
        # Temporal over last dimension iff temporal == True
        # Max Pooling after Con1D iff max_pool == True
        if temporal:
            y = TimeDistributed(Conv1D(features, kernel_size, stride))(x)
        else:
            y = Conv1D(features, kernel_size, stride)(x)            
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        if max_pool:
            y = MaxPool1D()(y)
        
        # LSTM Layer with BatchNormalization and Dropout
        y = LSTM(lstm_units,
                 #return_sequences=True,
                 recurrent_initializer='glorot_uniform',
                 stateful=True)(y)
        y = BatchNormalization()(y)
        if dropout > 0:
            y = Dropout(dropout)(y)
        
        # Dense Layer with Batch Normalization and Dropout
        if max_norm:
            kernel_constraint = tf.keras.constraints.MaxNorm(max_value=max_norm)
        else:
            kernel_constraint = None
        if dense_units > 0:
            y = Dense(dense_units, kernel_constraint = kernel_constraint)(y)
            y = BatchNormalization()(y)
            y = Activation('relu')(y)
            if dropout > 0:
                y = Dropout(dropout)(y)
        
        # Output Layer
        y = Dense(1)(y)
        
        # Full Model
        self.model = Model(inputs = x, outputs = y)
        
    def call(self, x):
        return self.model(x)
    
    def get_prediction(self, x, steps = 1):
        
        # reset model states
        self.model.reset_states()
        
        # initiate array for predictions
        predictions = np.zeros(steps)
        
        # initiate model using x and grab first prediction
        pred = self.model(x)
        pred = pred[:,-1:,:]
        predictions[0] = pred.numpy().reshape(1)[0]
        
        # iteratively generate next steps
        for i in range(1, steps):
            pred = self.model(pred)
            predictions[i] = pred.numpy().reshape(1)[0]
        
        return predictions
        

# Define DataSequence Generator Class
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class DataSequence(Sequence):
    def __init__(self, dataset, batch_size, seq_length, steps_per_epoch = 1,
                 weighted = False, return_sequences = True):
        self.dataset = dataset
        self.dataset_len = len(dataset)
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.steps_per_epoch = steps_per_epoch
        self.weighted = weighted
        self.return_sequences = return_sequences
        self.xshape = (batch_size, seq_length, self.dataset[0].shape[-1])
        if return_sequences:
            self.yshape = (batch_size, seq_length, 1)
        else:
            self.yshape = (batch_size, 1)
        self.dataset_lens = np.zeros(self.dataset_len, dtype = np.int32)
        for i in range(self.dataset_len):
            self.dataset_lens[i] = dataset[i].shape[0] - seq_length

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, epoch_step):
        
        # Initiate arrays for batches
        x_batch , y_batch = np.zeros(self.xshape) , np.zeros(self.yshape)
        
        # Randomly pick samples
        samples = np.random.choice(self.dataset_len, self.batch_size)
        
        # Randomly pick starting indx for each sample sequence and grab them
        indx_limits = self.dataset_lens[samples]
        for i in range(self.batch_size):
            sample_indx = np.random.choice(indx_limits[i])
            x_batch[i,:,:] = self.dataset[samples[i]][sample_indx:sample_indx+self.seq_length,:]
            # y_batch is shifted sequence or next single value
            if self.return_sequences:
                y_batch[i,:,:] = self.dataset[samples[i]][sample_indx+1:sample_indx+self.seq_length+1,0:1]
            else:
                y_batch[i,:] = self.dataset[samples[i]][sample_indx+1:sample_indx+2,0:1]
        
        # if weighted, weight each sample by reciprical of avg. price
        if self.weighted:
            return x_batch, y_batch, 1./x_batch.mean(axis = 1).mean(axis = 1)
        else:
            return x_batch, y_batch
    

# Function For Loading and Preparing Training/Validation Data
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def load_training(data_folder, col_names = ['5. adjusted close'],
                  max_date = None, validation_ratio = .0, lmbda = None):
    
    # grab all the filenames from the folder
    filenames = glob.glob(data_folder+'*')
    random.shuffle(filenames)
    
    # initiate list for storing data
    training_dataset = []
    
    # load in each file iteratively
    for filename in filenames:
        current_data = pd.read_csv(filename, encoding = 'UTF-8')
        # filter to dates we want
        if max_date:
            current_data =  current_data[current_data['date'] <= max_date]
        # filter columns we want
        current_data =  np.flip(current_data[col_names].to_numpy(), axis = 0)
        # Box-Cox Transform data if lmbda given
        if lmbda:
            current_data = BoxCox(current_data, lmbda = lmbda)
        # append to datasets
        training_dataset.append(current_data)
    
    # split into training/validation if needed and return dataset(s)
    if validation_ratio > 0.:
        n = len(training_dataset)
        m = int(n * validation_ratio)
        validation_dataset = training_dataset[-m:]
        training_dataset = training_dataset[:n-m]
        return np.array(training_dataset) , np.array(validation_dataset)
    else:
        return np.array(training_dataset)
    








    