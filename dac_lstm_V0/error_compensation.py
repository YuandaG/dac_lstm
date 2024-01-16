#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 16:57:41 2021

@author: adam
"""
import numpy as np
import pylab
import pandas as pd
import operator
import csv
import matplotlib.pyplot as plt
import statistics

r = '/Volumes/Extreme SSD/save model data/for peak load file/5min_350_features_30minfor_valley_batch_real.csv'
p = '/Volumes/Extreme SSD/save model data/Peak load/5min_350_features_30minfor_valley_batch.csv'
f = '/Volumes/Extreme SSD/save model data/for peak load file/5min_350_features_30minfor_valley_batch_fore.csv'
f_m = '/Volumes/Extreme SSD/save model data/for peak load file/5min_350_features_30minfor_valley_batch_fore.csv'

r = '/Volumes/Extreme SSD/save model data/household comparison/30household_365_5min_real.csv'
p = '/Volumes/Extreme SSD/save model data/Peak load/30household_365_5min.csv'
f = '/Volumes/Extreme SSD/save model data/household comparison/30household_365_5min_fore.csv'
f_m = '/Volumes/Extreme SSD/save model data/household comparison/30household_365_5min_fore.csv'


signal = 2

time_interval = 5

if time_interval == 30:
    real = pd.read_csv(r, header=0)
    peak = pd.read_csv(p, header=0)
    forecast = pd.read_csv(f, header=0)
    forecast_modify = pd.read_csv(f_m, header=0)
elif time_interval == 5:
    real = pd.read_csv(r, header=0)
    peak = pd.read_csv(p, header=0)
    forecast = pd.read_csv(f, header=0)
    forecast_modify = pd.read_csv(f_m, header=0)
    
peak_trigger = peak.values
forecast = forecast.values
forecast_modify = forecast_modify.values
real = real.values


# moving average smooth
def MA(dataset):
    forecast_mv = np.zeros((dataset.shape[0]+4,dataset.shape[1]))
    print(forecast_mv.shape[0])
    for i in range(forecast_mv.shape[0]):
        for j in range(forecast_mv.shape[1]):
            if i <= 1:
                forecast_mv[i,j] = dataset[i,j]
            elif i >= forecast_mv.shape[0]-2:
                # print('end')
                forecast_mv[i,j] = dataset[i-4,j]
            else:
                forecast_mv[i,j] = dataset[i-2,j]
    # plt.figure(dpi=150)
    # # plt.plot(forecast[-48*6*3:-48*6:,0],color = 'green', label='LSTM')
    # # plt.plot(forecast_mv[-48*6*3-2:-48*6-2,0], color = 'red', label='DAW-LSTM')
    # plt.plot(forecast[:48*6*3,0],color = 'green', label='LSTM')
    # plt.plot(forecast_modify[2:48*6*3+2,0], color = 'red', label='DAW-LSTM')
    # plt.legend()
    # plt.show()
            
    forecast_mv_av = np.zeros((dataset.shape[0]+4,dataset.shape[1]))
    for j in range(dataset.shape[1]):
        for i in range(dataset.shape[0]):
            forecast_mv_av[i+2,j] = (forecast_mv[i,j] + forecast_mv[i+1,j] + forecast_mv[i+2,j] + forecast_mv[i+3,j] + forecast_mv[i+4,j])/5
    forecast_modify[:,:] = forecast_mv_av[2:-2,:]
    return forecast_modify

# plt.figure(dpi=150)
# plt.plot(forecast[-48*6*3:-48*6:,0],color = 'green', label='LSTM')
# plt.plot(forecast_modify[-48*6*3:-48*6,0], color = 'red', label='DAW-LSTM')
# plt.legend()
# plt.show()



f = MA(forecast)
r = MA(real)
fm = MA(forecast)

pos = np.arange(1,2)
neg = np.arange(1,2)
forecast_step = forecast.shape[1]
step = np.arange(0,forecast_step)
# step = np.arange(0,6)
step = [5]
# steps = 1
coe = np.arange(1,2)
x = 0
y = 0
improve_count = np.zeros((forecast_step,len(pos),len(neg)),dtype=float)
savecsv='coefficient.csv'
import os
if os.path.exists(savecsv):  # 如果文件存在
    # 删除文件，可使用以下两种方法。
    os.remove(savecsv) 


for k in coe:
    for steps in step:  
        print(steps)
        for i in pos:
            for j in neg:
        
                # if time_interval == 30:
                #     real = pd.read_csv(r, header=0)
                #     peak = pd.read_csv(p, header=0)
                #     forecast = pd.read_csv(f, header=0)
                #     forecast_modify = pd.read_csv(f_m, header=0)
                # elif time_interval == 5:
                #     real = pd.read_csv(r, header=0)
                #     peak = pd.read_csv(p, header=0)
                #     forecast = pd.read_csv(f, header=0)
                #     forecast_modify = pd.read_csv(f_m, header=0)
                # k = forecast.shape[0]
                
        
                # peak_trigger = peak.values
                # forecast = forecast.values
                # forecast_modify = forecast_modify.values
                # real = real.values
                # peak_trigger = peak_trigger[-k+2:,0]
                        
                        
                #check 'peak_detection file for parameters' -----lag
                count = np.arange(48*1*6, len(peak_trigger))
                # count = np.arange(0, len(peak_trigger))
                
                estimation = []
                for num in count:
                    if peak_trigger[num,] != signal:
                        error = forecast[num,steps-1] - real[num,steps-1]
                        if abs(error) <= 20000:
                            estimation.append(error)
                
                 
                
                abs_estimation = list(map(abs, estimation)) 
                sum_est = statistics.mean((abs_estimation))
                estimation=pd.DataFrame(estimation)
                estimation=estimation.values
                # plt.plot(estimation[-48:,0])
                # plt.show()
                
                # for num in count:
                #     if num != count[-1,]:
                #         # print('start')
                #         if peak_trigger[num-1,] == 1:
                #             for a in np.arange(0,forecast_step):
                #                 # print(a)
                #                 error = forecast_modify[num-1+a,steps-1] - real[num-1+a,steps-1]
                #                 # print(error)
                #                 forecast_modify[num+a,steps-1] = forecast_modify[num+a,steps-1] - error
                
                
        
        
                # print(i,j)
                f = forecast
                r = real
                fm = forecast_modify
                for num in count:
                    # print(num)
                    if num != count[-forecast_step-1,]:
                        # print('start')
                        if peak_trigger[num,] != signal:
                            if peak_trigger[num,] == 1:
                                p = 0.8
                                n = 0.65
                                # p = 1
                                # n = 1
                            elif peak_trigger[num,] == -1:
                                p = 1.2
                                n = 1.2
                            else:
                                p = 0.95
                                n = 0.95
                            for a in np.arange(0,forecast_step):
                                # print(a)
                                error = f[num,0] - r[num,0]
                                if abs(error) <= 20000:
                                    # print(error)
                                    if error >= 0:               
                                        fm[num+1,a] = fm[num+1,a] - error/p
                                    elif error < 0:
                                        fm[num+1,a] = fm[num+1,a] - error/n
                                        # fm[num+1,a] = 0
                    else:
                        break
                            
                            
                estimation_modify = []
                for num in count:
                    if peak_trigger[num,] != signal:
                        error = fm[num,steps-1] - r[num,steps-1]
                        if abs(error) <= 20000:
                            estimation_modify.append(error)
        
                abs_estimation_modify = list(map(abs, estimation_modify))
                sum_est_modi = statistics.mean(abs_estimation_modify)
                
                improve = (sum_est - sum_est_modi)/sum_est
                improve_count[steps-1,x,y] = improve
                y += 1
            x += 1
            y = 0
        x = 0
        
        
estimation_modify=pd.DataFrame(estimation_modify)
estimation_modify=estimation_modify.values
# improve_count=pd.DataFrame(improve_count)
# improve_count.to_csv(savecsv,index=None)
position = 5

# print(improve_count)
plt.figure(dpi=150)
plt.plot(estimation[-48*6*5:-48*6*2:,0],color = 'green', label='LSTM')
plt.plot(estimation_modify[-48*6*5:-48*6*2,0], color = 'red', label='DAW-LSTM')
plt.legend()
plt.show()
plt.figure(dpi=150)
plt.plot(real[-48*6*5:-48*6*3,position],color = 'red', label='real')
plt.plot(forecast[-48*6*5:-48*6*3,position], color = 'green', label='forecast')
plt.plot(fm[-48*6*5:-48*6*3,position], color = 'blue', label='correction')
plt.legend()
plt.show()
plt.figure(dpi=150)
plt.plot(peak_trigger[-48*6*5:-48*6*3], color = 'blue', label='correction')
plt.legend()
plt.show()

k_av = 1 - ((forecast_modify- real) ** 2).sum() / ((forecast_modify - real.mean()) ** 2).sum()
print(k_av)
k2_av = 1 - ((forecast - real) ** 2).sum() / ((forecast- real.mean()) ** 2).sum()
print(k2_av)
ma_av = np.mean(abs((forecast_modify - real) / real))
print('Test MAPE: %.3f' % ma_av)

k = 1 - ((forecast_modify[:,position] - real[:,position]) ** 2).sum() / ((forecast_modify[:,position] - real[:,position].mean()) ** 2).sum()
print(k)
k2 = 1 - ((forecast[:,position] - real[:,position]) ** 2).sum() / ((forecast[:,position] - real[:,position].mean()) ** 2).sum()
print(k2)
ma1 = np.mean(abs((forecast[:,position] - real[:,position]) / real[:,position]))
print('Test MAPE: %.3f' % ma1)
ma2 = np.mean(abs((forecast_modify[:,position] - real[:,position]) / real[:,position]))
print('Test MAPE: %.3f' % ma2)
