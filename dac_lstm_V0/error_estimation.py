

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
from scipy.stats import pearsonr

# try:
#     time_interval=int(input('time interval：'))
#     # steps=int(input('forecast steps：'))
# except:
#     print("输入有误，请重新输入，输入数字，如：30")
#     time_interval=int(input('请输入时间间隔（键入回车键完成输入）：'))
from parameter import *


# GSPs, Basic, households, days
days_for_model =  550
household = 220


GSP_index = 'A1'
rolling = 4
if GSPs == 1:
    r = 'G:/LSTM Data/GSP_result/GSP_5min_%s_real.csv'%GSP_index
    p = 'G:/LSTM Data/GSP_peak/GSP_5min_%s_real.csv'%GSP_index
    f = 'G:/LSTM Data/GSP_result/GSP_5min_%s_forecast.csv'%GSP_index
    f_m = 'G:/LSTM Data/GSP_result/GSP_5min_%s_forecast.csv'%GSP_index
    print('GSP')
elif GSPs == 0 & long_test == 0 & household != 0 :
    r = 'G:/model_win/household comparison/%ihousehold_%i_5min_real.csv'%(household,days_for_model)
    p = 'G:/save model data/Peak load/%ihousehold_%i_5min.csv'%(household,days_for_model)
    f = 'G:/model_win/household comparison/%ihousehold_%i_5min_forecast.csv'%(household,days_for_model)
    f_m = 'G:/model_win/household comparison/%ihousehold_%i_5min_forecast.csv'%(household,days_for_model)
    print('household')
elif long_test == 1:
    r = 'G:/model_win/TV_general/%ihr_smooth_%ihousehold_%idays.csv'%(hr, household,days_for_model)
    p = 'G:/model_win/TV_general/peak/%ihr_forecast_%ihousehold_%idays.csv'%(hr,household,days_for_model)
    f = 'G:/model_win/TV_general/%ihr_forecast_%ihousehold_%idays.csv'%(hr,household,days_for_model)
    f_m = 'G:/model_win/TV_general/%ihr_forecast_%ihousehold_%idays.csv'%(hr,household,days_for_model)
    print(hr,household,days_for_model)
    print('TVVP')
else:
    print('Adjust parameters: GSP, household')

# household = 150
# r = '/Volumes/Extreme SSD/save model data/household comparison/%ihousehold_365_5min_real.csv'%household
# p = '/Volumes/Extreme SSD/save model data/Peak load/%ihousehold_365_5min.csv'%household
# f = '/Volumes/Extreme SSD/save model data/household comparison/%ihousehold_365_5min_fore.csv'%household
# f_m = '/Volumes/Extreme SSD/save model data/household comparison/%ihousehold_365_5min_fore.csv'%household


# # future scenario
# r = 'G:/LSTM Data/TravelPattern/2040_real.csv'
# p = 'G:/LSTM Data/TravelPattern/2040_peak.csv'
# f = 'G:/LSTM Data/TravelPattern/2040_forecast.csv'
# f_m = 'G:/LSTM Data/TravelPattern/2040_forecast.csv'



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

r=real.rolling(rolling,min_periods=1).mean()
f=forecast.rolling(rolling,min_periods=1).mean()
fm=forecast_modify.rolling(rolling,min_periods=1).mean()

r = r.values
f = f.values
fm = fm.values

peak_trigger = peak.values
forecast = forecast.values
forecast_modify = forecast_modify.values
real = real.values


# moving average smooth
# def MA(dataset):
#     forecast_mv = np.zeros((dataset.shape[0]+4,dataset.shape[1]))
#     print(forecast_mv.shape[0])
#     for i in range(forecast_mv.shape[0]):
#         for j in range(forecast_mv.shape[1]):
#             if i <= 1:
#                 forecast_mv[i,j] = dataset[i,j]
#             elif i >= forecast_mv.shape[0]-2:
#                 # print('end')
#                 forecast_mv[i,j] = dataset[i-4,j]
#             else:
#                 forecast_mv[i,j] = dataset[i-2,j]
#     # plt.figure(dpi=150)
#     # # plt.plot(forecast[-48*6*3:-48*6:,0],color = 'green', label='LSTM')
#     # # plt.plot(forecast_mv[-48*6*3-2:-48*6-2,0], color = 'red', label='DAW-LSTM')
#     # plt.plot(forecast[:48*6*3,0],color = 'green', label='LSTM')
#     # plt.plot(forecast_modify[2:48*6*3+2,0], color = 'red', label='DAW-LSTM')
#     # plt.legend()
#     # plt.show()
            
#     forecast_mv_av = np.zeros((dataset.shape[0]+4,dataset.shape[1]))
#     for j in range(dataset.shape[1]):
#         for i in range(dataset.shape[0]):
#             forecast_mv_av[i+2,j] = (forecast_mv[i,j] + forecast_mv[i+1,j] + forecast_mv[i+2,j] + forecast_mv[i+3,j] + forecast_mv[i+4,j])/5
#     forecast_modify[:,:] = forecast_mv_av[2:-2,:]
#     return forecast_modify

# plt.figure(dpi=150)
# plt.plot(forecast[-48*6*3:-48*6:,0],color = 'green', label='LSTM')
# plt.plot(forecast_modify[-48*6*3:-48*6,0], color = 'red', label='DAW-LSTM')
# plt.legend()
# plt.show()



# f = MA(forecast)
# r = MA(real)
# fm = MA(forecast)

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
                count = np.arange(48*1*1, len(peak_trigger))
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
                # f = forecast
                # r = real
                # fm = forecast_modify
                for num in count:
                    # print(num)
                    if num != count[-forecast_step-1,]:
                        # print('start')
                        if peak_trigger[num,] != signal:
                            if peak_trigger[num,] == 1:
                                p = 1
                                n = 1
                                # p = 1
                                # n = 1
                            # elif peak_trigger[num,] == -1:
                            #     p = 1
                            #     n = 1
                            else:
                                p = 1
                                n = 1
                            for a in np.arange(0,forecast_step):
                                # print(a)
                                error = f[num,2] - r[num,2]
                                if abs(error) <= 2000000:
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
a = 1
b = 0
c = 40
d = -1
# print(improve_count)
# plt.figure(dpi=100)
# plt.plot(estimation[-48*6*5:-48*6*2:,0],color = 'green', label='LSTM')
# plt.plot(estimation_modify[-48*6*5:-48*6*2,0], color = 'red', label='DAW-LSTM')
# plt.legend()
# plt.show()
plt.figure(dpi=300)
plt.plot(r[-48*6*a+c:-48*6*b+d,0],color = 'red', label='real')
# plt.plot(forecast[-48*6*a+c:-48*6*b+d,position], color = 'green', label='forecast')
# plt.plot(forecast[-48*6*a+c:-48*6*b+d,position], color = 'black', label='forecast')
# plt.plot(forecast[-48*6*a+c:-48*6*b+d,3], color = 'blue', label='forecast')
plt.plot(fm[-48*6*a+c:-48*6*b+d,position], color = 'blue', label='correction w/o cap')
plt.plot(f[-48*6*a+c:-48*6*b+d,position], color = 'purple', label='correction with cap')
plt.xlabel('Sample')
plt.ylabel('Power(W)')
plt.legend()


position = 5
a = 18
b = a-1
c = 0
d = 0

t = 1

fig, ax = plt.subplots(dpi=500)
ax.plot(r[-48*6*a+c:-48*6*b+d,4],color = 'red', label='Real',linewidth=t)
# ax.plot(forecast[-48*6*a+c:-48*6*b+d,position], color = 'green', label='Forecast')
ax.plot(f[-48*6*a+c:-48*6*b+d,position], color = 'green', label='ARIMA',linewidth=t)
ax.plot(fm[-48*6*a+c:-48*6*b+d,position], color = 'blue', label='DAC-LSTM',linewidth=t)
ax.plot(r[-48*6*a+c-6:-48*6*b+d-6,4], color = 'black', label='Persistence',linewidth=t)

ax.set_xticks(np.arange(0,48*6+1,24))
ax.set_xticklabels(np.arange(0,25,2))

# ax.set_xticks(np.arange(0,48*6*7+1,144))
# ax.set_xticklabels(np.arange(0,169,12))
plt.xlabel('Time (hours)') # 纵坐标轴标题
plt.ylabel('Power (W)') # 纵坐标轴标题
plt.legend(loc=2, prop={'size': 6})
plt.show()

# data = np.zeros((r.shape[0]-6,4))
# for i in range(r.shape[0]-6):
#     # data[i,0] = r[-48*6*a+c+6*i,0]
#     # data[i,1] = f[-48*6*a+c+6*i,5]
#     # data[i,2] = fm[-48*6*a+c+6*i,5]
#     # data[i,3] = r[-48*6*a+c+6*i-6,0]
    
#     data[i,0] = r[i,0]
#     data[i,1] = f[i,5]
#     data[i,2] = fm[i,5]
#     data[i,3] = r[i+6,0]
# plt.plot(data[:,0], label='Real')
# plt.plot(data[:,1], label='LSTM')
# plt.plot(data[:,2], label='DAC-LSTM')
# plt.plot(data[:,3], label='per')
# plt.legend()
# plt.show()

# savedsm=pd.DataFrame(data)
# savecsv4='G:/LSTM Data/TravelPattern/dsm_year.csv'
# savedsm.to_csv(savecsv4,index=None)  
# plt.show()
# plt.figure(dpi=100)
# plt.plot(peak_trigger[-48*6*a+c:-48*6*b+d], label='trigger')
# plt.legend()
# plt.show()


# plt.figure(dpi=150)
# fig = plt.figure(dpi=300)
# gs = fig.add_gridspec(2, hspace=0)
# axs = gs.subplots(sharex=True, sharey=False)
# axs[0].plot(r[-48*6*a+c:-48*6*b+d,position],color = 'red', label='real')
# axs[0].plot(forecast[-48*6*a+c:-48*6*b+d,position], color = 'green', label='forecast')
# axs[0].plot(fm[-48*6*a+c:-48*6*b+d,position], color = 'blue', label='correction')
# axs[1].plot(peak_trigger[-48*6*a+c:-48*6*b+d], label='trigger')
# plt.legend()
# plt.show()
# print('number of household: ',household)


r_av = 1 - ((fm- r) ** 2).sum() / ((fm - r.mean()) ** 2).sum()
print('m:',r_av)
r2_av = 1 - ((forecast - real) ** 2).sum() / ((forecast- real.mean()) ** 2).sum()
print(r2_av)
ma_av1 = np.mean(abs((fm - real) / real))
print('Test MAPE fm: %.4f' % ma_av1)
ma_av2 = np.mean(abs((forecast - real) / real))
print('Test MAPE: %.4f' % ma_av2)

# k = 1 - ((fm[:,position] - real[:,position]) ** 2).sum() / ((fm[:,position] - real[:,position].mean()) ** 2).sum()
# print('m:',k)
# k2 = 1 - ((forecast[:,position] - real[:,position]) ** 2).sum() / ((forecast[:,position] - real[:,position].mean()) ** 2).sum()
# print(k2)
# ma1 = np.mean(abs((fm[:,position] - real[:,position]) / real[:,position]))
# print('Test MAPE fm: %.3f' % ma1)
# ma2 = np.mean(abs((forecast[:,position] - real[:,position]) / real[:,position]))
# print('Test MAPE: %.3f' % ma2)

corr, _ = pearsonr(fm[:,5], r[:,1])
print('Test m Cor: %.3f' % corr)
corr, _ = pearsonr(f[:,5], r[:,1])
print('Test Cor: %.3f' % corr)
# k = 142
# plt.figure(dpi=150)
# plt.plot(real[k,:],color = 'red', label='real')
# plt.plot(forecast[k,:], color = 'green', label='forecast')
# plt.plot(fm[k,:], color = 'blue', label='correction')
# plt.xlabel('Sample')
# plt.ylabel('Power(W)')
# plt.legend()
# plt.show()

# table = np.zeros((48,6))
# # table.header = ['R2_m','R2','MAPE_m','MAPE','Corr_m','Corr']
# for i in np.arange(table.shape[0]):
#     # i = i+1
#     print(i)
#     j = 0
#     table[i,j] = 1 - ((fm[:,i]- r[:,i]) ** 2).sum() / ((fm[:,i] - r[:,i].mean()) ** 2).sum()
#     j = j+1
#     table[i,j] = 1 - ((forecast[:,i] - real[:,i]) ** 2).sum() / ((forecast[:,i]- real[:,i].mean()) ** 2).sum()
#     print('R2')
#     j = j+1
#     table[i,j] = np.mean(abs((fm[:,i] - real[:,i]) / real[:,i]))
#     j = j+1
#     table[i,j] = np.mean(abs((forecast[:,i] - real[:,i]) / real[:,i]))
#     print('MAPE')
#     print(j)
#     j = j+1
#     table[i,j], _ = pearsonr(fm[:,i], r[:,i])
#     j = j+1
#     table[i,j], _ = pearsonr(f[:,i], r[:,i])
#     print('corr')

# aaaaa4=pd.DataFrame(table)
# print(aaaaa4.header)
# aaaaa4.columns = ['R2_m','R2','MAPE_m','MAPE','Corr_m','Corr']
# savecsv4='G:/LSTM Data/TravelPattern/2040_daclstm_table.csv'
# aaaaa4.to_csv(savecsv4,index=None)    

    
    
    
    
    
    
    
    