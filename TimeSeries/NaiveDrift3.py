# -*- coding: utf-8 -*-
"""
Created on Tue May  5 23:17:25 2020

@author: Mitchell
"""

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
### Imports
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import NaiveDrift, ResidualAnalysis, ValidationSplit


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
### Naive Drift Model
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('Building Naive Drift Model\n--------------------------')

# Grab Data from Dataframe
#~~~~~~~~~~~~~~~~~~~~~~~~~
close = np.flip(VTI['5. adjusted close'].to_numpy())
datetime = np.flip(VTI.index.to_numpy())
time = np.int32((datetime - datetime[0]) / np.timedelta64(1, 'D'))

# Fill In Missing Days
#~~~~~~~~~~~~~~~~~~~~~
close_series = pd.Series(close)
close_series.index = datetime
datetime = pd.date_range(datetime[0], datetime[-1])
close = np.zeros(len(datetime))
close_series = close_series.reindex(datetime, fill_value = np.nan)
for i in range(len(close_series)):
    if np.isnan(close_series.iloc[i]):
        close[i] = close[i-1]
    else:
        close[i] = close_series.iloc[i]

# Split Data into Training and Validation
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
validation_ratio = .2
training_close , validation_close = ValidationSplit(close, validation_ratio)
training_datetime , validation_datetime = ValidationSplit(datetime, validation_ratio)
validation_size = len(validation_close)

# Full Model (Using All Data)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~
full_model = NaiveDrift(close)

# Test Model (Using Training Data)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
test_model = NaiveDrift(training_close)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
### Full Model Residuals
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Get Residuals
#~~~~~~~~~~~~~~
residuals = full_model.get_residuals()

# Residual Anlysis
#~~~~~~~~~~~~~~~~~
ResidualAnalysis(datetime[1:], residuals, nlags = 365)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
### Model Validation
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('Running Model Validation\n------------------------')

# Get Model Predictions
#~~~~~~~~~~~~~~~~~~~~~~
pred_close , pred_std = test_model.get_predictions(steps = validation_size,
                                                   return_std = True)

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
plt.fill_between(validation_datetime, pred_close-pred_std, pred_close+pred_std,
                 color='r', alpha = .4, label = '68% Conf. Interval')
plt.fill_between(validation_datetime, pred_close-2*pred_std, pred_close+2*pred_std,
                 color='r', alpha = .2, label = '95% Conf. Interval')
plt.legend()
plt.title('Naive Drift Model Prediction Test')
plt.xlabel('Date')
plt.ylabel('VTI Closing Price Per Share ($)')
plt.show()

# Print Statistics
#~~~~~~~~~~~~~~~~~
print('Mean and Std. of validation error: mu = {:6.4f}, sigma = {:6.4f}'\
      .format(err_mu, err_sigma))
accuracy_68 = ((validation_close >= pred_close-pred_std) *\
               (validation_close <= pred_close+pred_std)
               ).sum() / validation_size * 100
accuracy_95 = ((validation_close >= pred_close-2*pred_std) *\
               (validation_close <= pred_close+2*pred_std)
               ).sum() / validation_size * 100
mean_z_score = (error / pred_std).mean()
print('% of days actual price in 68% Confidence Interval: {:4.2f}%'\
      .format(accuracy_68))
print('% of days actual price in 95% Confidence Interval: {:4.2f}%'\
      .format(accuracy_95))
print('Average z-score of prediction error: {:3.2f}'.format(mean_z_score))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#----------------------------------END FILE------------------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~