# -*- coding: utf-8 -*-
"""
Created on Sun May  3 17:06:41 2020

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


### Alpha API Key
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
alpha_key_file = 'alpha_key.txt'
with open(alpha_key_file, 'r') as f:
    alpha_key = f.read()


### Ticker
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ticker = 'AVAV'


### TimeSeries Function (Pandas DF Output)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
av_time = TimeSeries(key = alpha_key, output_format = 'pandas')


### IntraDay Example
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
interval = '1min'
outputsize = 'full'
data_1, meta_data_1 = av_time.get_intraday(symbol = ticker,
                              interval = interval,
                              outputsize = outputsize)

print_data = True
if print_data:
    plt.figure()
    data_1['4. close'].plot()
    plt.title(ticker+' Intraday Closing Price by Minute')
    plt.xlabel('Day and Time')
    plt.ylabel('Price Per Share ($)')
    plt.show()


### IntraDay Adjusted Example
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
outputsize = 'full'
data_2, meta_data_2 = av_time.get_daily_adjusted(symbol = ticker,
                                                 outputsize = outputsize)

# Adjust for stock splits
N = len(data_2)
data_2['9. open adj_1'] = data_2['1. open']
data_2['10. high adj_1'] = data_2['2. high']
data_2['11. low adj_1'] = data_2['3. low']
data_2['12. close adj_1'] = data_2['4. close']
data_2['13. volume adj_1'] = data_2['6. volume']
data_2['14. dividend adj_1'] = data_2['7. dividend amount']
for i in range(N):
    split_value = data_2['8. split coefficient'].iloc[i]
    if split_value != 1.0:
        data_2['9. open adj_1'] = pd.concat((data_2['9. open adj_1'].iloc[:i+1], data_2['9. open adj_1'].iloc[i+1:]/split_value))
        data_2['10. high adj_1'] = pd.concat((data_2['10. high adj_1'].iloc[:i+1], data_2['10. high adj_1'].iloc[i+1:]/split_value))
        data_2['11. low adj_1'] = pd.concat((data_2['11. low adj_1'].iloc[:i+1], data_2['11. low adj_1'].iloc[i+1:]/split_value))
        data_2['12. close adj_1'] = pd.concat((data_2['12. close adj_1'].iloc[:i+1], data_2['12. close adj_1'].iloc[i+1:]/split_value))
        data_2['13. volume adj_1'] = pd.concat((data_2['13. volume adj_1'].iloc[:i+1], data_2['13. volume adj_1'].iloc[i+1:]/split_value))
        data_2['14. dividend adj_1'] = pd.concat((data_2['14. dividend adj_1'].iloc[:i+1], data_2['14. dividend adj_1'].iloc[i+1:]/split_value))

# Adjust for dividends
data_2['15. open adj_2'] = data_2['9. open adj_1']
data_2['16. high adj_2'] = data_2['10. high adj_1']
data_2['17. low adj_2'] = data_2['11. low adj_1']
data_2['18. close adj_2'] = data_2['12. close adj_1']
data_2['19. dividend adj_2'] = data_2['14. dividend adj_1']
for i in range(N):
    dividend = data_2['19. dividend adj_2'].iloc[i]
    if dividend != 0.0 and i != N-1:
        prev_close = data_2['18. close adj_2'].iloc[i+1]
        adj_ratio = (prev_close - dividend) / prev_close
        data_2['15. open adj_2'] = pd.concat((data_2['15. open adj_2'].iloc[:i+1], data_2['15. open adj_2'].iloc[i+1:]*adj_ratio))
        data_2['16. high adj_2'] = pd.concat((data_2['16. high adj_2'].iloc[:i+1], data_2['16. high adj_2'].iloc[i+1:]*adj_ratio))
        data_2['17. low adj_2'] = pd.concat((data_2['17. low adj_2'].iloc[:i+1], data_2['17. low adj_2'].iloc[i+1:]*adj_ratio))
        data_2['18. close adj_2'] = pd.concat((data_2['18. close adj_2'].iloc[:i+1], data_2['18. close adj_2'].iloc[i+1:]*adj_ratio))
        data_2['19. dividend adj_2'] = pd.concat((data_2['19. dividend adj_2'].iloc[:i+1], data_2['19. dividend adj_2'].iloc[i+1:]*adj_ratio))


print_data = True
if print_data:
    plt.figure()
    data_2['4. close'].plot(label = 'Close', linewidth = 1)
    data_2['12. close adj_1'].plot(label = 'Close Adj. Splits', linewidth = 1)
    data_2['18. close adj_2'].plot(label = 'Close Adj. Splits + Div.', linewidth = 1)
    plt.legend()
    plt.title(ticker+' Daily Closing Price')
    plt.xlabel('Date')
    plt.ylabel('Price Per Share ($)')
    plt.show()
    
    plt.figure()
    np.log(data_2['6. volume']).plot(label = 'Volume', linewidth = 1)
    np.log(data_2['13. volume adj_1']).plot(label = 'Volume Adj.', linewidth = 1)
    plt.legend()
    plt.title(ticker+' Daily Trading Volume')
    plt.xlabel('Date')
    plt.ylabel('Log of Trading Volume')
    plt.show()


### Save File
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
savefile = False
if savefile:
    today = str(date.today())
    save_folder = '../Data/'
    save_filename = save_folder + ticker + '-TimeSeriesDaily-' + today + '.txt'
    data_2.to_csv(save_filename, encoding = 'UTF-8')


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#----------------------------------END FILE------------------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~