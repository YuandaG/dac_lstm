# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 15:59:41 2021

@author: 18636
"""

# evaluate an ARIMA model using a walk-forward validation
from pandas import read_csv
from pandas import DataFrame as df
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from parameter import *
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")
import statsmodels.tsa.api as smt




# R2
def R2(y_test, y_true):
    return 1 - ((y_test - y_true) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum()

#MAPE
def mape(y_test, y_true):
    return np.mean(abs((y_test - y_true) / y_true))


rolling = 1


def main_ARIMA():
# load dataset

    # GSP, Basic, households, days
    # if GSPs == 1:
    #     # household = 0
    #     loaddata = read_csv('G:/LSTM Data/GSP2011_5min_%s.csv'%GSP_index, header=0)
    #     loaddata = -1*loaddata.rolling(rolling,min_periods=1).mean()
    #     loaddata=pd.DataFrame(loaddata)
    #     loaddata = loaddata.values
    #     dataset = np.zeros((loaddata.shape[0],loaddata.shape[1]))
    #     dataset[:,0] = loaddata[:,0]
    #     print('GSP')
    # # #For household
    # elif household != 0 & GSPs == 0:
    #     if long_test == 1:
    #         loaddata = read_csv('G:/model_win/TV_general/%ihr_smooth_%ihousehold_%idays.csv'%(hr, household,days_for_model))
    #         print('05hr')
    #     elif load_test == 0:
    #         loaddata = read_csv('G:/model_win/household comparison/%ihousehold_%i_5min_real.csv'%(household,days_for_model), header=0)
    #         loaddata = loaddata['0'].rolling(rolling,min_periods=1).mean()
    #         print('5min')
    #     loaddata = pd.DataFrame(loaddata)
    #     loaddata = loaddata.values  
    #     x = loaddata[:,0]
    #     dataset = np.zeros((loaddata.shape[0],loaddata.shape[1]))
    #     dataset[:,0] = loaddata[:,0]
    #     # dataset[:,:] = loaddata[:,:]
    #     # dataset[:,1] = loaddata[:,1]
    #     print('TVVP household')
        
    # elif household == 0 & GSPs == 0:
    #     loaddata = read_csv('G:/LSTM/load_error_5min.csv', header=0)
    #     dataset = np.zeros((loaddata.shape[0],loaddata.shape[1]))
    #     loaddata = loaddata.values
    #     dataset[:,0] = loaddata[:,0]+evdata[:,0]
    #     dataset[:,1] = loaddata[:,1]
    #     print('TVVP')
    # else:
    #     print('Adjust parameters: GSP, household')
        
    # print(household, days_for_model)
    
    # future scenario
    # loaddata = read_csv('G:/LSTM Data/TravelPattern/2040_real.csv', header=0)
    loaddata = read_csv('G:/model_win/household comparison/220household_550_5min_real.csv', header=0)
    dataset = np.zeros((loaddata.shape[0],1))
    loaddata = loaddata.values
    # loaddata = loaddata[:,0]
    # find lag value
    def tsplot(y, lags=None, figsize=(12, 7)):
        """
            Plot time series, its ACF and PACF, calculate Dickey–Fuller test
            
            y - timeseries
            lags - how many lags to include in ACF, PACF calculation
        """
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
               
        fig = plt.figure(figsize=figsize)
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (1, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (2, 0))
        pacf_ax = plt.subplot2grid(layout, (2, 1))
    
        y.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()
        
    # plt.plot(x)
    x = loaddata[:,0]

    # for item in range(len(x)):
    #     if x[item] >=100000 or x[item] <=-30000:
    #         # print(item)
    #         x[item] = x[item-1]
    #     else:
    #         x[item] = x[item]
    # # k = x      
    # #Original
    # x = pd.Series(x)
    # tsplot(x)
    # #first order
    # ts_sun_diff = (x - x.shift(1)).dropna()
    # tsplot(ts_sun_diff)
    # # plt.plot(ts_sun_diff)
    # #second order
    # sec_diff = (ts_sun_diff - ts_sun_diff.shift(1)).dropna()
    # tsplot(sec_diff)
    # # third order
    # third_diff = (sec_diff - sec_diff.shift(1)).dropna()
    # tsplot(third_diff)
    
    # for_diff = (third_diff - third_diff.shift(1)).dropna()
    # tsplot(for_diff)
    
    
    

    
    # values = values[values.index % 6 == 0]
    X = x
    # split into train and test sets
    
    size = int(len(X) * 0.66)
    train, test = X[66:size], X[size:len(X)]
    
    repeat = test.shape[0]-47
    repeat = 48*6*2
    # prediction = np.zeros((6, repeat))
    print('starts')
    # for i in range(test.shape[0]-47):
    for i in range(repeat):
        train = X[66+i:size+i]
        history = [x for x in train]
        # model = AutoARIMA(history)
        model = ARIMA(history, order=(4,2,4))
        model_fit = model.fit()
        start_index = len(history)
        end_index = start_index + 5
        forecast = model_fit.predict(start=start_index, end=end_index)
        forecast = forecast.T
        prediction[:,i] = forecast[:]
        print(i)
    
    
    real = np.zeros((6, repeat))
    for i in range(prediction.shape[0]):
        for j in range(prediction.shape[1]):
            real[i,j] = test[j+i]
    # real = real.T
    
    
    
    
    
    # predictions = list()
    # # walk-forward validation
    # print('starts')
    # test_y = test[:,0]
    # for t in range(len(test)):
    #     model = ARIMA(history, order=(19,1,0) )
    #     model_fit = model.fit()
    #     output = model_fit.forecast(step = 48)
    #     yhat = output[0]
    #     print(output)
    #     predictions.append(yhat)
    #     obs = test[t]
    #     history.append(obs)
    #     if t != 0:
    #         corr,_ = pearsonr(np.array(predictions), test_y[0:len(predictions)])
    #         print('predicted=%f, expected=%f, %i/%i, R2=%.3f, MAPE=%.4f, Cor=%.3f' % (yhat, 
    #                                                                                   obs, 
    #                                                                                   t, 
    #                                                                                   len(test), 
    #                                                                                   R2(np.array(predictions),test_y[0:len(predictions)]), 
    #                                                                                   mape(np.array(predictions),test_y[0:len(predictions)]), 
    #                                                                                   corr))
    
    # for t in range(len(test)):
    # 	model = ARIMA(history, order=(5,1,0))
    # 	model_fit = model.fit()
    # 	output = model_fit.forecast()
    # 	yhat = output[0]
    # 	predictions.append(yhat)
    # 	obs = test[t]
    # 	history.append(obs)
    # 	print('predicted=%f, expected=%f, %i/%i, R2=%.3f, MAPE=%.3f, Cor=%.3f' % (yhat, obs, t, len(test), R2(np.array(predictions),test[0:len(predictions)]), mape(np.array(predictions),test[0:len(predictions)]), pearsonr(np.array(predictions),test[0:len(predictions)])))
        
    
    print(household, days_for_model)
    
    # evaluate forecasts
    # rmse = sqrt(mean_squared_error(real, prediction))
    rmse = sqrt(mean_squared_error(test, prediction))
    print('Test RMSE: %.3f' % rmse)
    # plot forecasts against actual outcomes
    # pyplot.plot(test)
    # pyplot.plot(predictions, color='red')
    # pyplot.show()
    
    
    R2_score = R2(prediction, real)
    mape_score = mape(prediction, real)
    b = list()
    for i in range(real.shape[1]):
        k, _ = pearsonr(prediction[:,i], real[:,i])
        b.append(k)
    b = np.array(k)
    corr = b.sum().mean()
    # corr, _ = pearsonr(prediction, real)
    
    print('Test R2: %.3f' % R2_score)
    print('Test MAPE: %.3f' % mape_score)
    print('Test Cor: %.3f' % corr)
    return rmse, R2_score, mape_score, corr


day = [14,30,50,100,150,200,250,300,350,400,450,500,550]
house = [15,30,50,100,150,220]

# housedata = np.zeros((6,4))
# daydata = np.zeros((13,4)) 

# a = 0
# for item in house:
#     household = item
#     days_for_model = 550
#     rmse, R2_score, mape_score, corr = main_ARIMA()
#     housedata[a,0] = rmse
#     housedata[a,1] = R2_score
#     housedata[a,2] = mape_score
#     housedata[a,3] = corr
#     a = a + 1
    
# a = 0

# for item in day:
#     household = 220
#     days_for_model = item
#     rmse, R2_score, mape_score, corr = main_ARIMA()   
#     daydata[a,0] = rmse
#     daydata[a,1] = R2_score
#     daydata[a,2] = mape_score
#     daydata[a,3] = corr
#     a = a + 1


def persistence():
    # # GSP, Basic, households, days
    # if GSPs == 1:
    #     # household = 0
    #     loaddata = read_csv('G:/LSTM Data/GSP2011_5min_%s.csv'%GSP_index, header=0)
    #     loaddata = -1*loaddata.rolling(rolling,min_periods=1).mean()
    #     loaddata=pd.DataFrame(loaddata)
    #     loaddata = loaddata.values
    #     dataset = np.zeros((loaddata.shape[0],loaddata.shape[1]))
    #     dataset[:,0] = loaddata[:,0]
    #     print('GSP')
    # # #For household
    # elif household != 0 & GSPs == 0:
    #     if long_test == 1:
    #         loaddata = read_csv('G:/model_win/TV_general/%ihr_smooth_%ihousehold_%idays.csv'%(hr, household,days_for_model))
    #         print('05hr')
    #     elif load_test == 0:
    #         loaddata = read_csv('G:/model_win/household comparison/%ihousehold_%i_5min_real.csv'%(household,days_for_model), header=0)
    #         loaddata = loaddata['0'].rolling(rolling,min_periods=1).mean()
    #         print('5min')
    #     loaddata = pd.DataFrame(loaddata)
    #     loaddata = loaddata.values  
    #     x = loaddata[:,0]
    #     dataset = np.zeros((loaddata.shape[0],loaddata.shape[1]))
    #     dataset[:,0] = loaddata[:,0]
    #     # dataset[:,:] = loaddata[:,:]
    #     # dataset[:,1] = loaddata[:,1]
    #     print('TVVP household')
        
    # elif household == 0 & GSPs == 0:
    #     loaddata = read_csv('G:/LSTM/load_error_5min.csv', header=0)
    #     dataset = np.zeros((loaddata.shape[0],loaddata.shape[1]))
    #     loaddata = loaddata.values
    #     dataset[:,0] = loaddata[:,0]+evdata[:,0]
    #     dataset[:,1] = loaddata[:,1]
    #     print('TVVP')
    # else:
    #     print('Adjust parameters: GSP, household')
        
    # print(household, days_for_model)
    
    
    # future scenario
    loaddata = read_csv('G:/LSTM Data/TravelPattern/2040_real.csv', header=0)
    dataset = np.zeros((loaddata.shape[0],1))
    loaddata = loaddata.values
    dataset = loaddata
    
    series = dataset
    values = df(series[:,0])
    X = values.values
    size = int(len(X) * 0.66)
    real = X[6:,0]
    prediction = X[:-6,0]
    R2_score = R2(prediction, real)
    mape_score = mape(prediction, real)
    # covariance = cov(prediction, real)
    corr, _ = pearsonr(prediction, real)
    # test_score = mean_squared_error(test_y, predictions)
    print('Test R2: %.3f' % R2_score)
    print('Test MAPE: %.4f' % mape_score)
    print('Test Cor: %.3f' % corr)
    
    if household == 15:
        plt.plot(real)
        plt.show()
    return R2_score, mape_score, corr

housedata_per = np.zeros((6,3))
daydata_per = np.zeros((13,3)) 

a = 0
for item in house:
    household = item
    days_for_model = 550
    R2_score, mape_score, corr = persistence()
    # housedata_per[a,0] = rmse
    housedata_per[a,0] = R2_score
    housedata_per[a,1] = mape_score
    housedata_per[a,2] = corr
    a = a + 1

    
a = 0

for item in day:
    household = 220
    days_for_model = item
    R2_score, mape_score, corr = persistence()
    # daydata_per[a,0] = rmse
    daydata_per[a,0] = R2_score
    daydata_per[a,1] = mape_score
    daydata_per[a,2] = corr
    a = a + 1

aaaaa4=pd.DataFrame(housedata_per)
savecsv4='G:/model_win/TV_general/data/per_house.csv'
aaaaa4.to_csv(savecsv4,index=None)