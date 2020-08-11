# -*- coding: utf-8 -*-
"""
Created on Mon May  4 00:10:48 2020

@author: Mitchell
"""

### Imports
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import alpha_vantage as av
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import time


### Alpha API Key
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
alpha_key_file = 'alpha_key.txt'
with open(alpha_key_file, 'r') as f:
    alpha_key = f.read()


### Tickers
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#tickers_file = 'S&P500-Tickers.csv'
tickers_file = 'Russell3000.csv'
tickers = list(pd.read_csv(tickers_file, encoding = 'UTF-8')['Symbol'])


### TimeSeries Function (Pandas DF Output)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
av_time = TimeSeries(key = alpha_key, output_format = 'pandas')


### Function for Loading and Saving Single Stock
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def LoadnSaveStock(ticker, save_folder):
    # load data using alpha_vantage api
    outputsize = 'full'
    data, meta_data = av_time.get_daily_adjusted(symbol = ticker,
                                                 outputsize = outputsize)
    # Adjust for stock splits
    N = len(data)
    data['9. open adj_1'] = data['1. open']
    data['10. high adj_1'] = data['2. high']
    data['11. low adj_1'] = data['3. low']
    data['12. close adj_1'] = data['4. close']
    data['13. volume adj_1'] = data['6. volume']
    data['14. dividend adj_1'] = data['7. dividend amount']
    for i in range(N):
        split_value = data['8. split coefficient'].iloc[i]
        if split_value != 1.0:
            data['9. open adj_1'] = pd.concat((data['9. open adj_1'].iloc[:i+1], data['9. open adj_1'].iloc[i+1:]/split_value))
            data['10. high adj_1'] = pd.concat((data['10. high adj_1'].iloc[:i+1], data['10. high adj_1'].iloc[i+1:]/split_value))
            data['11. low adj_1'] = pd.concat((data['11. low adj_1'].iloc[:i+1], data['11. low adj_1'].iloc[i+1:]/split_value))
            data['12. close adj_1'] = pd.concat((data['12. close adj_1'].iloc[:i+1], data['12. close adj_1'].iloc[i+1:]/split_value))
            data['13. volume adj_1'] = pd.concat((data['13. volume adj_1'].iloc[:i+1], data['13. volume adj_1'].iloc[i+1:]/split_value))
            data['14. dividend adj_1'] = pd.concat((data['14. dividend adj_1'].iloc[:i+1], data['14. dividend adj_1'].iloc[i+1:]/split_value))
    
    # Adjust for dividends
    data['15. open adj_2'] = data['9. open adj_1']
    data['16. high adj_2'] = data['10. high adj_1']
    data['17. low adj_2'] = data['11. low adj_1']
    data['18. close adj_2'] = data['12. close adj_1']
    data['19. dividend adj_2'] = data['14. dividend adj_1']
    for i in range(N):
        dividend = data['19. dividend adj_2'].iloc[i]
        if dividend != 0.0 and i != N-1:
            prev_close = data['18. close adj_2'].iloc[i+1]
            adj_ratio = (prev_close - dividend) / prev_close
            data['15. open adj_2'] = pd.concat((data['15. open adj_2'].iloc[:i+1], data['15. open adj_2'].iloc[i+1:]*adj_ratio))
            data['16. high adj_2'] = pd.concat((data['16. high adj_2'].iloc[:i+1], data['16. high adj_2'].iloc[i+1:]*adj_ratio))
            data['17. low adj_2'] = pd.concat((data['17. low adj_2'].iloc[:i+1], data['17. low adj_2'].iloc[i+1:]*adj_ratio))
            data['18. close adj_2'] = pd.concat((data['18. close adj_2'].iloc[:i+1], data['18. close adj_2'].iloc[i+1:]*adj_ratio))
            data['19. dividend adj_2'] = pd.concat((data['19. dividend adj_2'].iloc[:i+1], data['19. dividend adj_2'].iloc[i+1:]*adj_ratio))
    
    # save to csv file
    today = str(date.today())
    save_filename = save_folder + ticker + '-TimeSeriesDaily-' + today + '.txt'
    data.to_csv(save_filename, encoding = 'UTF-8')
    print('Saving Daily TimeSeries for {} to file: {}'.format(ticker, save_filename))
    

### Load in S&P 500 Stocks
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#save_folder = '../Data/S&P500/'
save_folder = '../Data/Russell3000/'
for ticker in tickers:
    # handle ValueErrors for invlaid tickers
    try:
        LoadnSaveStock(ticker, save_folder)
    except ValueError:
        print('Failed to Load TimeSeries Data for {}'.format(ticker))
    time.sleep(15)
    

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#----------------------------------END FILE------------------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~