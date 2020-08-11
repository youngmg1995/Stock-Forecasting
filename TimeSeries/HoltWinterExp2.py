# -*- coding: utf-8 -*-
"""
Created on Fri May  8 20:57:43 2020

@author: Mitchell
"""

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
### Imports
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing
from scipy import stats
from utils import BoxCox, InvBoxCox, NaiveDrift, ResidualAnalysis,\
    SeasonalAdd, ValidationSplit


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
### Load Data
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Read Data from CSV
#~~~~~~~~~~~~~~~~~~~
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


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
### Transform Data
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Grab Data from Dataframe
#~~~~~~~~~~~~~~~~~~~~~~~~~
close = np.flip(VTI['5. adjusted close'].to_numpy())
datetime = np.flip(VTI.index.to_numpy())
time = np.int32((datetime - datetime[0]) / np.timedelta64(1, 'D'))

# Box-Cox Transform
#~~~~~~~~~~~~~~~~~~
log_close = BoxCox(close)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
### Holt-Winter's Model
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("Building Holt-Winter's Model\n"+'-'*28)

# Split Data into Training and Validation
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
validation_ratio = .2
training_close , validation_close = ValidationSplit(log_close, validation_ratio)
training_datetime , validation_datetime = ValidationSplit(datetime, validation_ratio)
validation_size = len(validation_close)

# Full Model (Using All Data)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~
period = 20
full_model = ExponentialSmoothing(log_close, trend = True#, seasonal = period
                                  ).fit(maxiters = 100000)

# Test Model (Using Training Data)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
test_model = ExponentialSmoothing(training_close, trend = True
                                  #seasonal = period
                                  ).fit(maxiters = 100000)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
### Full Model Residuals
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Get Residuals
#~~~~~~~~~~~~~~
residuals = log_close - full_model.predict(0, len(close)-1)

# Residual Anlysis
#~~~~~~~~~~~~~~~~~
ResidualAnalysis(datetime, residuals, nlags = 252)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
### Model Validation
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('Running Model Validation\n------------------------')

# Get Model Predictions
#~~~~~~~~~~~~~~~~~~~~~~
pred_close = test_model.forecast(validation_size)

# Get Erros
#~~~~~~~~~~
error = validation_close - pred_close
err_mu , err_sigma = error.mean() , error.std()

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
print('Mean and Std. of validation error: mu = {:6.4f}, sigma = {:6.4f}'\
      .format(err_mu, err_sigma))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#----------------------------------END FILE------------------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~