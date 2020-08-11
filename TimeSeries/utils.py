# -*- coding: utf-8 -*-
"""
Created on Wed May  6 21:24:31 2020

@author: Mitchell
"""

### Imports
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa import stattools


### Naive Drift Model Class
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class NaiveDrift(object):
    '''
    Class used for defining our naive drift forecasting model.
    '''
    def __init__(self, y):
        '''
        Instatiates a new NaiveDrift model instance using the given dataset.
        Basically stores the dataset, calculates the drift term, and gets the
        residuals.
        '''
        self.y = y
        self.drift = (y[-1] - y[0]) / (len(y) - 1)
        self.residuals = (y[1:] - y[:-1]) - self.drift
    
    def get_predictions(self, steps = 1, return_std = False):
        '''
        Calcualtes predictions for next x steps. Will also return the standard
        deviation of the predictions if return_std set to True.
        '''        
        # get predictions
        h = np.arange(1,steps+1)
        pred = self.y[-1] + self.drift*h
        
        # get standard deviations of predictions if desired
        if return_std:
            T = len(self.y)
            std = self.residuals.std() * np.sqrt(h*(1+h/T))
            
            return pred , std
        
        else:
            
            return pred

    def get_residuals(self):
        '''
        Returns residuals of model.
        '''
        return self.residuals


### Naive Drift Model Class
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class NaiveDrift2(object):
    '''
    Class used for defining our naive drift forecasting model.
    '''
    def __init__(self, t, y):
        '''
        Instatiates a new NaiveDrift model instance using the given dataset.
        Basically stores the dataset, calculates the drift term, and gets the
        residuals.
        '''
        self.y = y
        self.t = t
        self.drift = (y[-1] - y[0]) / (t[-1] - t[0])
        self.residuals = (y[1:] - y[:-1]) - self.drift*(t[1:] - t[:-1])
    
    def get_predictions(self, t, return_std = False):
        '''
        Calcualtes predictions for next x steps. Will also return the standard
        deviation of the predictions if return_std set to True.
        '''        
        # get predictions
        h = t - self.t[-1:]
        pred = self.y[-1] + self.drift*h
        
        # get standard deviations of predictions if desired
        if return_std:
            T = len(self.y)
            std = self.residuals.std() * np.sqrt(h*(1+h/T))
            
            return pred , std
        
        else:
            
            return pred

    def get_residuals(self):
        '''
        Returns residuals of model.
        '''
        return self.residuals


### Add Back Seasonal Component Function
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def SeasonalAdd(x, seasonal, period):
    '''
    Function that adds back seasonal component to x. To do so, takes the last
    period from the seasonal vector as the seasonal component.
    '''
    # get length of x
    N = len(x)
    
    # grab periodic seasonal compontent
    ys = seasonal[-period:]
    
    # add sesonal component to x
    y = np.zeros(N)
    for i in range(N):
        j = i % period
        y[i] = x[i] + ys[j]
    
    return y


### Residual Analysis Function
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def ResidualAnalysis(x, residuals,  nbins = 200, nlags = 365, missing = False):
    '''
    Runs analysis of model residuals. To do so, plots residuals, histogram of
    residuals with fitted normal distribution, and ACF (Autocorrelation
    Coefficients) of residuals. User has ability to set number of bins for
    histogram and maximum lag considered for ACF. Also prints some of these
    coefficients.
    '''
    # Initiate Figure
    #~~~~~~~~~~~~~~~~
    fig = plt.figure(figsize=(8, 8))
    ax2 = fig.add_subplot(2,2,3)
    ax3 = fig.add_subplot(2,2,4)
    ax1 = fig.add_subplot(2,1,1)
    
    # Plot 1) Residuals
    #~~~~~~~~~~~~~~~~~~
    ax1.plot(x, residuals, linewidth = 1)
    ax1.set_title('Residuals for Naive Drfit Model')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Residual Value')
    
    # Plot 2) Residual Distribution
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # histogram
    n, bins, patches = ax2.hist(residuals, nbins, density=1, facecolor='blue', alpha=0.5, label = 'Residuals')
    # add normal distr. for comparison
    mu , sigma = residuals.mean() , residuals.std()
    normal_distr = NormDistr(bins, mu, sigma)
    ax2.plot(bins, normal_distr, 'r', label = 'Fit Normal Distr.')
    # add normal distr. with outliers removed
    ed_res, outliers = RemoveOutliers(residuals)
    ed_mu , ed_sigma = ed_res.mean() , ed_res.std()
    ed_normal_distr = NormDistr(bins, ed_mu, ed_sigma)
    ax2.plot(bins, ed_normal_distr, 'g--', label = 'Outliers Removed')
    # labels, notations, legend etc.
    ax2.legend()
    ax2.set_title('Histogram of Model Residuals')
    ax2.set_xlabel('Residual Value')
    ax2.set_ylabel('Normalized Frequency (Prob.)')
    
    # Plot 3) Residual Autocorrelation Coefficients (ACF)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    acfs = stattools.acf(residuals, nlags = nlags, fft = True)[1:]
    n = len(acfs)
    T = len(residuals)
    # plot ACF        
    ax3.bar(range(1, n+1), acfs)
    ax3.plot(range(1, n+1), np.ones(n)*2/np.sqrt(T), 'r--')
    ax3.plot(range(1, n+1), -np.ones(n)*2/np.sqrt(T), 'r--')
    ax3.set_title('ACF of Residuals')
    ax3.set_xlabel('Lag (Days)')
    ax3.set_ylabel('Autocorrelation Coefficient')
    
    
    # Plot Figure
    #~~~~~~~~~~~~
    #fig.tight_layout()
    fig.show()
    
    # Print Statistics
    #~~~~~~~~~~~~~~~~~
    print('Mean and Std. of residuals: mu = {:6.4f}, sigma = {:6.4f}'\
          .format(mu, sigma))
    print('Mean and Std. with outliers removed: mu = {:6.4f}, sigma = {:6.4f}'\
          .format(ed_mu, ed_sigma))
    print('Number of outliers removed: {} = {:4.2f}% of residuals\n'\
          .format(len(outliers), len(outliers)/len(residuals)*100))
       

### Validatio Split Function
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def ValidationSplit(dataset, validation_ratio):
    '''
    Splits dataset into training and validation sets according to given ratio.
    '''
    validation_size = int(len(dataset)*validation_ratio)
    training_size = len(dataset) - validation_size
    training = dataset[:training_size]
    validation = dataset[-validation_size:]
    
    return training , validation


### Box-Cox Transformation
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def BoxCox(x, lmbda = 0):
    '''
    Performs Box-Cox Transformation on our dataset shown below:
        
        y = ln(x)                  for lmbda = 0
            (x**lmbda - 1)/lmbda    for lmbda != 0
    '''
    if lmbda == 0:
        y = np.log(x)
    else:
        y = (x**lmbda - 1.) / lmbda
    
    return y


### Inverse Box-Cox Transformation
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def InvBoxCox(x, lmbda = 0, bias_adjust = False, sigma = None):
    '''
    Performs inverse of Box-Cox Transformation on our dataset shown below:
        
        y = ln(x)                  for lmbda = 0
            (x**lmbda - 1)/lmbda    for lmbda != 0
    
    Technically, if the input is an array of means, then the output in the
    median. To account for this we allow bias_adjust = True to be specified,
    which then returns an adjusted transformation using the given sigmas. This
    debiased output will be the mean of the distribution rather than the
    median.
    '''
    if bias_adjust:
        if lmbda == 0:
            y = np.exp(x) * (1 + (sigma**2)/2)
        else:
            y = (lmbda*x + 1.)**(1./lmbda) * (1+((1-lmbda)*(sigma**2))/(2*(lmbda*x+1)**2))
    else:
        if lmbda == 0:
            y = np.exp(x)
        else:
            y = (lmbda*x + 1.)**(1./lmbda)
    
    return y


### Normal Distribution Function
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def NormDistr(x, mu, sigma):
    '''
    Calculates values for each point in array x as a function of a Normal
    distribution with mean mu and standard deviation sigma.
    '''
    return np.exp(-(x-mu)**2 / (2*sigma**2)) / (sigma * (2*np.pi)**(1/2))


### Likelyhood Function
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def Likelyhood(x, mu, sigma):
    '''
    Calculates likelihood of samples x given the assumption that they are
    drawn from a normal distribution with mean mu and standard deviation sigma.
    '''
    n = len(x)
    return (2*np.pi*sigma**2)**(-n/2) * np.exp(-1/(2*sigma**2) * ((x-mu)**2).sum())


### Log-Likelyhood Function
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def LogLikelyhood(x, mu, sigma):
    '''
    Calculates loglikelihood of samples x given the assumption that they are
    drawn from a normal distribution with mean mu and standard deviation sigma.
    '''
    n = len(x)
    return -n/2*np.log(2*np.pi*sigma**2) - 1/(2*sigma**2)*((x-mu)**2).sum()


### Outlier Removal Function
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def RemoveOutliers(x):
    '''
    Removes outliers using same method taught in grade school. That is, any
    values more than 1.5 interquartile distances from the inner quartiles is
    considered an outlier. Iteratively remove outliers using this method.
    '''
    # initiate first outlier check
    q1 , q3 = np.quantile(x, [.25, .75])
    iqd = q3 - q1
    new_x = x[(x >= (q1 - 1.5*iqd)) * (x <= (q3 + 1.5*iqd))]
    new_outliers = x[(x < (q1 - 1.5*iqd)) + (x > (q3 + 1.5*iqd))]
    outliers = new_outliers
    x = new_x
    still_outliers = len(new_outliers) > 0
    while still_outliers:
        q1 , q3 = np.quantile(x, [.25, .75])
        iqd = q3 - q1
        new_x = x[(x >= (q1 - 1.5*iqd)) * (x <= (q3 + 1.5*iqd))]
        new_outliers = x[(x < (q1 - 1.5*iqd)) + (x > (q3 + 1.5*iqd))]
        outliers = np.concatenate((outliers, new_outliers))
        x = new_x
        still_outliers = len(new_outliers) > 0
    
    return x , outliers
