#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 13:52:14 2021

@author: adam
"""
import matplotlib.pyplot as plt
import numpy as np
import pylab
import pandas as pd
import operator
import csv
from parameter import *

# savecsv='peak5min_uk_afterlockdown.csv'
# savecsv='/Volumes/Extreme SSD/save model data/Peak load/5min_550_features_30minfor_valley_batch.csv'




# GSP, Basic, households, days
# days_for_model =  14
# household = 220
# if long_test == 0:
#     savecsv='G:/save model data/Peak load/%ihousehold_%i_5min.csv'%(household,days_for_model)
# elif long_test == 1:
#     savecsv='G:/model_win/TV_general/peak/%ihr_forecast_%ihousehold_%idays.csv'%(hr,household,days_for_model)
    

# GSPs_index = 'A1'
# if long_test == 0 & GSPs == 1:
#     y = pd.read_csv('G:/LSTM Data/GSPs_result/GSPs_5min_%s_real.csv'%GSPs_index, header=0)
#     savecsv='G:/LSTM Data/GSPs_peak/GSPs_5min_%s_real.csv'%GSPs_index
#     print('GSPs')
# elif long_test == 0 & household != 0 & GSPs == 0:
#     y = pd.read_csv('G:/model_win/household comparison/%ihousehold_%i_5min_real.csv'%(household,days_for_model), header=0)
#     savecsv='G:/save model data/Peak load/%ihousehold_%i_5min.csv'%(household,days_for_model)
#     print('TVVP household')
# elif long_test == 1:
#     # y = pd.read_csv('G:/save model data/for peak load file/5min_350_features_30minfor_valley_batch_real.csv', header=0)
#     y = pd.read_csv('G:/model_win/TV_general/%ihr_smooth_%ihousehold_%idays.csv'%(hr, household,days_for_model))
#     savecsv='G:/model_win/TV_general/peak/%ihr_forecast_%ihousehold_%idays.csv'%(hr,household,days_for_model)
#     print('TVVP')
# else:
#     print('Adjust parameters: GSPs, household')

# print(household, days_for_model)
# import os
# if os.path.exists(savecsv):  # 如果文件存在
#     # 删除文件，可使用以下两种方法。
#     os.remove(savecsv)

# print(household, days_for_model)



# future scenario
y = pd.read_csv('G:/LSTM Data/TravelPattern/2040_real.csv', header=0)
savecsv='G:/LSTM Data/TravelPattern/2040_peak.csv'

def thresholding_algo(y, lag, threshold, influence):
    signals = np.zeros(len(y))
    filteredY = np.array(y)
    avgFilter = [0]*len(y)
    stdFilter = [0]*len(y)
    avgFilter[lag - 1] = np.mean(y[0:lag])
    stdFilter[lag - 1] = np.std(y[0:lag])
    for i in range(lag, len(y)):
        if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter [i-1]:
            if y[i] > avgFilter[i-1]:
                signals[i] = 1
            else:
                signals[i] = -1

            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])
        else:
            signals[i] = 0
            filteredY[i] = y[i]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])

    return dict(signals = np.asarray(signals),
                avgFilter = np.asarray(avgFilter),
                stdFilter = np.asarray(stdFilter))

# y = pd.read_csv('/Users/adam/Desktop/Pycharm/LSTM/30_60real_mul.csv', header=0, index_col=0)
# y = pd.read_csv('/Users/adam/Desktop/Pycharm/LSTM/5_30real_mul.csv', header=0, index_col=0)
# y = pd.read_csv('/Users/adam/Desktop/Pycharm/LSTM/5_30real_uk_afterlockdown.csv', header=0, index_col=0)
# y = pd.read_csv('/Users/adam/Desktop/Pycharm/LSTM/sum_of_ev_load.csv', header=0)
# y = pd.read_csv('/Volumes/Extreme SSD/save model data/household comparison/%ihousehold_365_5min_real.csv'%household, header=0)

# y = y['0']
y = y.values
# y = y[-48*7*2:,0]
y = y[:,0]

# Settings: lag = 30, threshold = 5, influence = 0
# lag = 48*7*6
lag = 48*1*1
threshold = 1
influence = 1

# Run algo with settings from above

result = thresholding_algo(y, lag=lag, threshold=threshold, influence=influence)
# shiftstdFilter = [0]*len(y)
# shiftavgFilter = [0]*len(y)
# shiftavgFilter[0:-lag+1] = result["avgFilter"][lag-1:]
# shiftstdFilter[0:-lag+1] = result["stdFilter"][lag-1:]
# k = np.add(shiftavgFilter + threshold * shiftstdFilter)list(map(operator.add, first,second))
# Plot result
plt.figure(dpi=200)
pylab.subplot(211)
pylab.plot(np.arange(1, len(y)+1), y)

# pylab.plot(np.arange(1, len(y)+1),
#            shiftavgFilter, color="cyan", lw=2)

# pylab.plot(np.arange(1, len(y)+1),
#            np.add(shiftavgFilter, threshold * shiftstdFilter), color="green", lw=2)


# pylab.plot(np.arange(1, len(y)+1),
#             np.add(shiftavgFilter, threshold * shiftstdFilter), color="green", lw=2)

pylab.plot(np.arange(1, len(y)+1),
            result["avgFilter"], color="cyan", lw=2, label='Average')

pylab.plot(np.arange(1, len(y)+1),
            result["avgFilter"] + threshold * result["stdFilter"], color="green", lw=2, label='Average_upperbond')


pylab.plot(np.arange(1, len(y)+1),
            result["avgFilter"] - threshold * result["stdFilter"], color="green", lw=2, label='Average_lowerbond')
plt.legend()
pylab.subplot(212)
pylab.step(np.arange(1, len(y)+1), result["signals"], color="red", lw=2, label='Signal')
pylab.ylim(-1.5, 1.5)
plt.legend()
pylab.show()

data = result["signals"]
# with open('output_path','w',newline='') as file:

#     write=csv.writer(file, delimiter='\n')
#     for num in data:
#         write.writerow(num)
data=pd.DataFrame(data)
data.to_csv(savecsv,index=None)