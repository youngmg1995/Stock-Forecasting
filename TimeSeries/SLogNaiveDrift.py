# -*- coding: utf-8 -*-
"""
Created on Fri May  8 14:38:40 2020

@author: Mitchell
"""

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
### Imports
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
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
### Naive Drift Model
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('Building Seasonal Naive Drift Model on Log Data\n'+'-'*47)

# Split Data into Training and Validation
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
validation_ratio = .2
training_close , validation_close = ValidationSplit(log_close, validation_ratio)
training_datetime , validation_datetime = ValidationSplit(datetime, validation_ratio)
validation_size = len(validation_close)

# Seasonal Decomposition
#~~~~~~~~~~~~~~~~~~~~~~~
period= 252
full_decomp = seasonal_decompose(log_close, period = period,
                                 extrapolate_trend = 'freq')
training_decomp = seasonal_decompose(training_close, period = period,
                                     extrapolate_trend = 'freq')

# Plot Full Decomposition
#~~~~~~~~~~~~~~~~~~~~~~~~
full_decomp.plot()

# Full Model (Using All Data)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~
full_model = NaiveDrift(full_decomp.trend + full_decomp.resid)

# Test Model (Using Training Data)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
test_model = NaiveDrift(training_decomp.trend + training_decomp.resid)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
### Full Model Residuals
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Get Residuals
#~~~~~~~~~~~~~~
residuals = full_model.get_residuals()

# Remove Log
#~~~~~~~~~~~
residuals = InvBoxCox(log_close[1:]) - InvBoxCox(log_close[1:] - residuals)

# Residual Anlysis
#~~~~~~~~~~~~~~~~~
ResidualAnalysis(datetime[1:], residuals, nlags = 252)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
### Model Validation
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('Running Model Validation\n------------------------')

# Get Model Predictions
#~~~~~~~~~~~~~~~~~~~~~~
log_pred_close , log_pred_std = test_model.get_predictions(steps = validation_size,
                                                           return_std = True)

# Add Back Seasonal Component
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~
log_pred_close = SeasonalAdd(log_pred_close, training_decomp.seasonal, period)

# Remove Log
#~~~~~~~~~~~
# data
training_close =  InvBoxCox(training_close) 
validation_close =  InvBoxCox(validation_close) 
# predictions
pred_close_median =  InvBoxCox(log_pred_close)
pred_close_mean =  InvBoxCox(log_pred_close, bias_adjust = True,
                             sigma = log_pred_std)
# uncertainty ranges
lower_68 =  InvBoxCox(log_pred_close - log_pred_std)
upper_68 =  InvBoxCox(log_pred_close + log_pred_std)
lower_95 =  InvBoxCox(log_pred_close - 2*log_pred_std)
upper_95 =  InvBoxCox(log_pred_close + 2*log_pred_std)

# Get Erros
#~~~~~~~~~~
abs_error = np.abs(validation_close - pred_close_mean)
ame , ame_std = abs_error.mean() , abs_error.std()

# Plot Predictions
#~~~~~~~~~~~~~~~~~
plt.figure()
plt.plot(training_datetime[-validation_size:], training_close[-validation_size:],
         'b', linewidth = 1, label = 'Training')
plt.plot(validation_datetime, validation_close, 'k', linewidth = 1,
         label = 'Validation')
plt.plot(validation_datetime, pred_close_mean, 'r', linewidth = 1,
         label = 'Prediction Mean')
plt.plot(validation_datetime, pred_close_median, 'r--', linewidth = 1,
         label = 'Prediction Median')
plt.fill_between(validation_datetime, lower_68, upper_68,
                 color='r', alpha = .4, label = '68% Conf. Interval')
plt.fill_between(validation_datetime, lower_95, upper_95,
                 color='r', alpha = .2, label = '95% Conf. Interval')
plt.legend()
plt.title('Naive Drift Model Prediction Test')
plt.xlabel('Date')
plt.ylabel('VTI Closing Price Per Share ($)')
plt.show()

# Print Statistics
#~~~~~~~~~~~~~~~~~
print('Mean and Std. of validation absolute error: mu = {:6.4f}, sigma = {:6.4f}'\
      .format(ame, ame_std))
accuracy_68 = ((validation_close >= lower_68) *\
               (validation_close <= upper_68)
               ).sum() / validation_size * 100
accuracy_95 = ((validation_close >= lower_95) *\
               (validation_close <= upper_95)
               ).sum() / validation_size * 100
mean_z_score = (np.abs((BoxCox(validation_close) - log_pred_close)) / log_pred_std).mean()
print('% of days actual price in 68% Confidence Interval: {:4.2f}%'\
      .format(accuracy_68))
print('% of days actual price in 95% Confidence Interval: {:4.2f}%'\
      .format(accuracy_95))
print('Average z-score of prediction error: {:3.2f}'.format(mean_z_score))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#----------------------------------END FILE------------------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~