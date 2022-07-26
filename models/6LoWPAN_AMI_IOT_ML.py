# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 16:20:39 2022
If  each  house hold is having an  Energy utility meter connected  over 6LOWPAN or other  Wireless 802.15.4 based /WISUN
based  utility network then by using the  "raw_data" which comes from the household power consumption how to do applied
AI/ML based analysis  to derive data insights is the focus of this ML project with IOT data.
@author: Akram Sheriff
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn
import seaborn as sns
from scipy.special import factorial

power = pd.read_csv('C:/Users/sheriff/.spyder-py3/AKRAM_CODE_FOLDER/ML/household_power_consumption.csv', parse_dates=['Date_Time'])
power = power.set_index('Date_Time')
power.head()

for i in list(power.columns):
    power = power[pd.to_numeric(power[i], errors='coerce').notnull()]

for i in list(power.columns):
    power[[i]] = power[[i]].astype('float32')

print(power.dtypes)

plt.figure(figsize=(14, 7))
voltage = power.loc['2007-01-18':'2007-01-26', ['Voltage']]
plt.plot(voltage.rolling(360, center=True).mean())

import statsmodels.api as sm
from statsmodels.tsa.api import ExponentialSmoothing

train = power.loc['2007-01-18':'2007-01-24', ['Voltage']]
test = power.loc['2007-01-25':'2007-01-26', ['Voltage']]
len(test)

sm.tsa.seasonal_decompose(train, freq = 360).plot()

train = power.loc['2007-01-18':'2007-01-24', ['Voltage']].resample('H').mean()
test = power.loc['2007-01-25':'2007-01-26', ['Voltage']].resample('H').mean()

fit1 = ExponentialSmoothing(np.asarray(train), seasonal_periods = 24, seasonal = 'add', trend = 'add' ).fit()

y_hat_avg = test.copy()
y_hat_avg['Holt_Winter'] = fit1.forecast(len(test))

plt.figure(figsize=(16,8))
plt.plot( train, label='Train')
plt.plot(test, label='Test')
plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')
plt.legend(loc='best')
plt.grid()
plt.savefig('Holt_Winter.png')
plt.show()

from dateutil.relativedelta import relativedelta # working with dates with style
from scipy.optimize import minimize              # for function minimization

import statsmodels.formula.api as smf            # statistics and econometrics
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs

from itertools import product                    # some useful functions
from tqdm import tqdm_notebook

import warnings                                  # `do not disturbe` mode
warnings.filterwarnings('ignore')

from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error

def mean_absolute_percentage_error(y_true, y_pred): 
    return (np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

def moving_average(series, n):
    """
        Calculate average of last n observations
    """
    return np.average(series[-n:])

moving_average(power.Voltage, 1440) # prediction for the last observed day (past 24 hours)

volts = power.loc['2007-01-18':'2007-01-26', ['Voltage']]
volts_hourly =power.loc['2007-01-18':'2007-01-26', ['Voltage']].resample('H').mean()
print(len(volts), len(volts_hourly))

plt.figure(figsize=(15, 7))
plt.plot(power.loc['2007-01-18':'2007-01-26', ['Voltage']])
plt.title('Voltage (minutely data)')
plt.grid(True)
plt.show()

def plotMovingAverage(series, window, plot_intervals=False, scale=1.96, plot_anomalies=False):

    """
        series - dataframe with timeseries
        window - rolling window size 
        plot_intervals - show confidence intervals
        plot_anomalies - show anomalies 

    """
    rolling_mean = series.rolling(window=window).mean()

    plt.figure(figsize=(15,5))
    plt.title("Moving average\n window size = {}".format(window))
    plt.plot(rolling_mean, "g", label="Rolling mean trend")

    # Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bond = rolling_mean - (mae + scale * deviation)
        upper_bond = rolling_mean + (mae + scale * deviation)
        plt.plot(upper_bond, "r--", label="Upper Bond / Lower Bond")
        plt.plot(lower_bond, "r--")
        
        # Having the intervals, find abnormal values
        if plot_anomalies:
            anomalies = pd.DataFrame(index=series.index, columns=series.columns)
            anomalies[series<lower_bond] = series[series<lower_bond]
            anomalies[series>upper_bond] = series[series>upper_bond]
            plt.plot(anomalies, "ro", markersize=10)
        
    plt.plot(series[window:], label="Actual values")
    plt.legend(loc="upper left")
    plt.grid(True)
    
plotMovingAverage(volts, 360, plot_intervals = True, plot_anomalies = True)

plt.figure(figsize=(15,5))
plt.plot(volts_hourly)
plt.title('Hourly Voltage')
plt.grid(True)
plt.savefig('volts_hourly.png')

plotMovingAverage(volts_hourly, 2)
plt.title('Hourly Voltage (moving average window = 2)')
plt.savefig('volts_hourly_avg.png')

plotMovingAverage(volts_hourly, 2, plot_intervals = True)
plt.title('Hourly Voltage (moving average window = 2)')
plt.savefig('volts_hourly_avg_intervals.png')

plotMovingAverage(volts_hourly, 2, plot_intervals = True, plot_anomalies = True)
plt.title('Hourly Voltage (moving average window = 2)')
plt.savefig('volts_hourly_avg_intervals_anomalies.png')

kitchen = power.loc['2007-01-18':'2007-01-26', ['Kitchen']]
kitchen_hourly =power.loc['2007-01-18':'2007-01-26', ['Kitchen']].resample('H').mean()
print(len(kitchen), len(kitchen_hourly))

plotMovingAverage(kitchen_hourly, 2, plot_intervals=True, plot_anomalies=True)
plt.title('Kitchen Hourly Output (moving average window = 2)')
plt.savefig('kitchen_hourly_avg_intervals_anomalies.png')







