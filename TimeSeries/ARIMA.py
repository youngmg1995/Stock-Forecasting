# -*- coding: utf-8 -*-
"""
Created on Fri May  8 21:47:48 2020

@author: Mitchell
"""

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
### Imports
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from utils import BoxCox, InvBoxCox, ResidualAnalysis, ValidationSplit
import itertools
import time as tm


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
### Grid Search for Model Parameters
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Set to True to Do Search
#~~~~~~~~~~~~~~~~~~~~~~~~~
grid_search = False
if grid_search:
    
    # Set Timer
    #~~~~~~~~~~
    tic = tm.perf_counter()
    
    # Set Parameters for Grid Search
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    p = d = q = range(5)
    pdq = list(itertools.product(p, d, q))
    #seasonal_pdq = [(x[0], x[1], x[2], 252) for x in list(itertools.product(p, d, q))]
    
    # Run Grid Search
    #~~~~~~~~~~~~~~~~
    min_ACI = 10^10
    for param in pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(log_close,
                                            order=param,
                                            #seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{} - AIC:{}'.format(param, results.aic))
            if results.mle_retvals['converged']:
                mod = sm.tsa.statespace.SARIMAX(log_close,
                                                order=param,
                                                #seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                results = mod.fit(cov_type = 'robust')
            if results.aic < min_ACI and results.mle_retvals['converged']:
                min_ACI = results.aic
                min_pdq = param
        except:
            print('Failed to fit ARIMA{}'.format(param))
            
    # Print Time
    #~~~~~~~~~~~
    toc = tm.perf_counter()
    print(f"\nGrid search completed in {(toc - tic)/60:0.2f} minutes.")
    print("Optimal Parameters: (p,d,q) = {}\n".format(min_pdq))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
### ARIMAX Model
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('Building Arima Model on Log Data\n'+'-'*32)

# Split Data into Training and Validation
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
validation_ratio = .2
training_close , validation_close = ValidationSplit(log_close, validation_ratio)
training_datetime , validation_datetime = ValidationSplit(datetime, validation_ratio)
validation_size = len(validation_close)

# Set pdq If No Grid Search
#~~~~~~~~~~~~~~~~~~~~~~~~~~
if not grid_search:
    min_pdq = (2, 0, 0)

# Full Model (Using All Data)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~
full_model = sm.tsa.statespace.SARIMAX(log_close,
                                       order=min_pdq,
                                       #seasonal_order=param_seasonal,
                                       enforce_stationarity=False,
                                       enforce_invertibility=False).fit()

# Test Model (Using Training Data)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
test_model = sm.tsa.statespace.SARIMAX(training_close,
                                       order=min_pdq,
                                       #seasonal_order=param_seasonal,
                                       enforce_stationarity=False,
                                       enforce_invertibility=False).fit()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
### Full Model Residuals
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Get Residuals
#~~~~~~~~~~~~~~
start = min_pdq[0]
residuals = InvBoxCox(log_close[start:]) -\
    InvBoxCox(full_model.predict(start, len(close)-1))

# Residual Anlysis
#~~~~~~~~~~~~~~~~~
ResidualAnalysis(datetime[2:], residuals, nlags = 252)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
### Model Validation
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('Running Model Validation\n------------------------')

# Get Model Predictions
#~~~~~~~~~~~~~~~~~~~~~~
pred = test_model.get_prediction(len(training_close), len(training_close)+validation_size-1)
log_pred_close = pred.predicted_mean
log_pred_std = pred.se_mean

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