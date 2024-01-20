

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
import math
from sklearn.preprocessing import MinMaxScaler
# try:
#     time_interval=int(input('time interval：'))
#     # steps=int(input('forecast steps：'))
# except:
#     print("输入有误，请重新输入，输入数字，如：30")
#     time_interval=int(input('请输入时间间隔（键入回车键完成输入）：'))
from parameter import *
import os
from mpl_toolkits import mplot3d

# days_for_model =  550
# household = 220
rolling = 1

# r = '/Volumes/Extreme SSD/save model data/household comparison/%ihousehold_365_5min_real.csv'%household
# p = '/Volumes/Extreme SSD/save model data/Peak load/%ihousehold_365_5min.csv'%household
# f = '/Volumes/Extreme SSD/save model data/household comparison/%ihousehold_365_5min_fore.csv'%household
# f_m = '/Volumes/Extreme SSD/save model data/household comparison/%ihousehold_365_5min_fore.csv'%household

r = os.path.join(path_5min,'%ihr_smooth_%ihousehold_%idays.csv'%(hr, household,days_for_model))
p = os.path.join(path_5min,'%ihr_peak_%ihousehold_%idays.csv'%(hr, household,days_for_model))
f = os.path.join(path_5min,'%ihr_forecast_%ihousehold_%idays.csv'%(hr, household,days_for_model))
f_m = os.path.join(path_5min,'%ihr_forecast_%ihousehold_%idays.csv'%(hr, household,days_for_model))

time_interval = 5

#TODO: check header=0?
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

r_ori=real.rolling(rolling,min_periods=1).mean()
f_ori=forecast.rolling(rolling,min_periods=1).mean()
fm_ori=forecast_modify.rolling(rolling,min_periods=1).mean()


r_ori = r_ori.values
r = r_ori
f_ori = f_ori.values
f = f_ori
fm_ori = fm_ori.values
fm = fm_ori

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


        
def alpha(i):
    alpha = (1.5 - 1/(1+e**(1-i)/2))
    return alpha

def beta(i):
    beta = (E_t/E_avg) * k * ((2/(5+5*e**(n*(1-E_t/E_avg)))-0.2))*((1-i)/m+1)
    # beta = (2/(5+5*(3)**(n)-2)((1-i)/m+1) 
    return beta

def R2(fm,r):
    r2 = 1 - ((fm- r) ** 2).sum() / ((fm - r.mean()) ** 2).sum()
    return r2
    
def MAPE(fm,r):
    mape = np.mean(abs((fm - r) / r))
    return mape

def e_avg(num):
    E = 0
    for j in range(avg_num):
        e_average = r[num-j-1,0] - f[num-j-1,0]
        E = E + e_average
    E_avg = E / avg_num
    return E_avg

def e_avg_1(num):
    E = 0
    for j in range(avg_num):
        e_average_1 = r[num-j-2,0] - f[num-j-2,0]
        E = E + e_average_1
    E_avg_1 = E / avg_num
    return E_avg_1

# def main():
#     k = 1
#     for num in count:
#         if num <= count[-forecast_step-1,]:
#             # peak
#             if peak_trigger[num,] == 1:
#                 E_t = r[num,5] - f[num,5]
#                 E_avg = e_avg(num)
#                 E_avg_1 = e_avg_1(num)
#                 # E_avg = (r[num-3,0] - f[num-3,0] + r[num-2,0] - f[num-2,0] + r[num-1,0] - f[num-1,0])/3
#                 # E_avg_1 = (r[num-4,0] - f[num-4,0] + r[num-3,0] - f[num-3,0] + r[num-2,0] - f[num-2,0])/3
#                 E_t_1 = f[num-1,5] - r[num-1,5]
#                 if E_t_1 > E_t:
#                     k = k - delta
#                 else:
#                     k = k + delta
#                 for i in np.arange(0,forecast_step):

#                     com = alpha(i) * beta(i,E_t,E_avg)
#                     if com >= volve:
#                         com = volve
#                     fm[num+1,i] = fm[num+1,i] + com * E_t
                    
#                     # if i == 0:
#                     #     k_acc.append(k)
#                     #     com_acc.append(com)
#                 if (E_t_1/E_avg_1) * (E_t/E_avg) <= 0:
#                     k = 1
                    
#             # off-peak
#             if peak_trigger[num,] != 1:
#                 E_t = r[num,5] - f[num,5]
#                 E_avg = e_avg(num)
#                 E_avg_1 = e_avg_1(num)
#                 # E_avg = (r[num-3,0] - f[num-3,0] + r[num-2,0] - f[num-2,0] + r[num-1,0] - f[num-1,0])/3
#                 # E_avg_1 = (r[num-4,0] - f[num-4,0] + r[num-3,0] - f[num-3,0] + r[num-2,0] - f[num-2,0])/3
#                 E_t_1 = f[num-1,5] - r[num-1,5]
#                 if E_t_1 > E_t:
#                     k = k - delta
#                 else:
#                     k = k + delta
                
#                 for i in np.arange(0,forecast_step):

#                     com = alpha(i) * beta(i,E_t,E_avg)
#                     if com >= volve:
#                         com = volve
#                     fm[num+1,i] = fm[num+1,i] + com * E_t
                    
#                     # if i == 0:
#                         # k_acc.append(k)
#                         # com_acc.append(com)
#                 if (E_t_1/E_avg_1) * (E_t/E_avg) <= 0:
#                     k = 1

#             # k_acc.append(k)
#             # com_acc.append(com)
#             # # alpha_acc.append(alpha(i))
#             # if beta(i) < 5:
                
#             #     beta_acc.append(beta(i))
            
#     return fm

# plt.plot(fm[:,5]-fm_ori[:,5])



forecast_step = forecast.shape[1]

m_start = 85
n_start = 30
m_number = 1
n_number = 1
avg_start = 221
avg_number = 1
# E_avg
avg_range = np.arange(avg_start,avg_start + avg_number)
m_range = np.arange(m_start,m_start + m_number)
n_range = np.arange(n_start,n_start + n_number)

k = 1
delta = 0.05
e = 2.718
volve = 1.2

count = np.arange(48*1*1, len(peak_trigger))
record_r2 = np.zeros((m_range.shape[0],n_range.shape[0],avg_range.shape[0]))
record_mape = np.zeros((m_range.shape[0],n_range.shape[0],avg_range.shape[0]))
k_acc = []
com_acc = []
alpha_acc = []
beta_acc = []


ahead = 5
print(R2(f[:,5],r[:,5]))
for avg_num in avg_range:
    for m in m_range:
        for n in n_range:
            # fm = forecast
            # f = forecast
            dac = np.zeros((forecast.shape[0],6))
            # r = r_ori
            print('reset')
            # dac = main()
            for num in count:
                if num <= count[-forecast_step-1,]:
                    # peak
                    if peak_trigger[num,] == 1:
                        E_t = r[num,ahead] - f[num,ahead]
                        E_avg = e_avg(num)
                        E_avg_1 = e_avg_1(num)
                        # E_avg = (r[num-3,0] - f[num-3,0] + r[num-2,0] - f[num-2,0] + r[num-1,0] - f[num-1,0])/3
                        # E_avg_1 = (r[num-4,0] - f[num-4,0] + r[num-3,0] - f[num-3,0] + r[num-2,0] - f[num-2,0])/3
                        E_t_1 = f[num-1,ahead] - r[num-1,ahead]
                        if E_t_1 > E_t:
                            k = k - delta
                        else:
                            k = k + delta
                        for i in np.arange(0,forecast_step):
    
                            com = alpha(i) * beta(i)*1
                            if com >= (volve * k):
                                com = (volve * k)
                            dac[num+1,i] = fm[num+1,i] + com * E_t
                            
                            # if i == 0:
                            #     k_acc.append(k)
                            #     com_acc.append(com)
                        if (E_t_1/E_avg_1) * (E_t/E_avg) <= 0:
                            k = 1
                            
                    # off-peak
                    if peak_trigger[num,] != 1:
                        E_t = r[num,ahead] - f[num,ahead]
                        E_avg = e_avg(num)
                        E_avg_1 = e_avg_1(num)
                        # E_avg = (r[num-3,0] - f[num-3,0] + r[num-2,0] - f[num-2,0] + r[num-1,0] - f[num-1,0])/3
                        # E_avg_1 = (r[num-4,0] - f[num-4,0] + r[num-3,0] - f[num-3,0] + r[num-2,0] - f[num-2,0])/3
                        E_t_1 = f[num-1,ahead] - r[num-1,ahead]
                        if E_t_1 > E_t:
                            k = k - delta
                        else:
                            k = k + delta
                        
                        for i in np.arange(0,forecast_step):
    
                            com = alpha(i) * beta(i)*1
                            if com >= (volve * k):
                                com = (volve * k)
                            dac[num+1,i] = fm[num+1,i] + com * E_t
                            
                            # if i == 0:
                                # k_acc.append(k)
                                # com_acc.append(com)
                        if (E_t_1/E_avg_1) * (E_t/E_avg) <= 0:
                            k = 1
       
                    k_acc.append(k)
                    com_acc.append(com)
                    alpha_acc.append(alpha(i))
                    # if beta(i) < 10000000000000:
                        
                    beta_acc.append(beta(i))
            # print(m,n)
            r2 = R2(dac,r)
            mape = MAPE(dac,r)
            record_r2[m-m_start,n-n_start,avg_num-avg_start] =  r2
            record_mape[m-m_start,n-n_start,avg_num-avg_start] =  mape

            print(m,n,R2(dac[:,0],r[:,0])) 
            print(m,n,R2(dac[:,5],r[:,5])) 
            print(m,n,R2(dac,r)) 
            

# eva_r2 = np.zeros((288,1))
# eva_r2[:,0] = record_r2[0,0,:]
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_r2 = scaler.fit_transform(eva_r2)  

# eva_mape = np.zeros((288,1))
# eva_mape[:,0] = record_mape[0,0,:]
# # scaler = MinMaxScaler(feature_range=(0, 4.04))
# # scaled_mape = scaler.fit_transform(eva_mape)  


# # plt.plot(scaled_r2)

# # plt.ylabel('Number of days to calculate E_avg')
# # plt.xlabel('R2')

         
# plt.plot(record_r2[0,0,:])

# x = np.arange(1, 289)

# fig = plt.figure(dpi=150)
# ax1 = fig.add_subplot(111)

# ax1.set_xlabel('Number of samples to calculate E_avg') 
# ax1.set_ylabel('R2') 
# lns1 = ax1.plot(x, scaled_r2, 'red', label='R2') 
# ax2 = ax1.twinx()
# ax2.set_ylabel('MAPE(%)') 
# lns2 = ax2.plot(x, eva_mape, 'green', label='MAPE') 

# # added these three lines
# lns = lns1+lns2
# labs = [l.get_label() for l in lns]
# ax1.legend(lns, labs, loc=0)





# max_r2 = np.max(record_r2)
#         # E_t = forecast[num,steps-1] - real[num,steps-1]
# eva = record_r2[:,:,0]

# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_values = scaler.fit_transform(eva)

# eva = record_r2[:,:,0]
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_values = scaler.fit_transform(eva)
# # plt.plot(eva[6,:])
# # plt.plot(scaled_values[:,6])

# eva = record_mape[:,:,0]*100
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_values = scaler.fit_transform(eva)

# m = np.arange(0,100)
# n = np.arange(0,100)

# X, Y = np.meshgrid(m, n)
# Z = eva
# fig = plt.figure(dpi=150)
# ax = plt.axes(projection='3d')
# # ax.plot_wireframe(X, Y, Z, color='black')
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
#                 cmap='viridis', edgecolor='none');
# # ax.scatter(X, Y, Z, c=Z, cmap='viridis', linewidth=1);
# ax.set_xlabel('m')
# ax.set_ylabel('n')
# ax.set_zlabel('MAPE(%)');
# ax.view_init(20, 20)




# saver2='r2_m%ito%i_n%ito%i_Eavg%i.csv'%(m_start,m_number,n_start,n_number,avg_start)
# if os.path.exists(saver2):  # 如果文件存在
#     # 删除文件，可使用以下两种方法。
#     os.remove(saver2) 
# r2_table=pd.DataFrame(eva)
# r2_table.to_csv(saver2,index=None)  

# savemape='mape_m%ito%i_n%ito%i_Eavg%i.csv'%(m_start,m_number,n_start,n_number,avg_start)
# if os.path.exists(savemape):  # 如果文件存在
#     # 删除文件，可使用以下两种方法。
#     os.remove(savemape) 
# r2_table=pd.DataFrame(eva)
# r2_table.to_csv(savemape,index=None)  





plt.plot(k_acc)
plt.show()
plt.plot(com_acc)
plt.show() 
plt.plot(alpha_acc)
plt.show() 
plt.plot(beta_acc)
plt.show() 
# estimation_modify=pd.DataFrame(estimation_modify)
# estimation_modify=estimation_modify.values
# improve_count=pd.DataFrame(improve_count)
# improve_count.to_csv(savecsv,index=None)

# plt.plot(fm[:,0]-f[:,0])
# plt.plot(-f[:,0]+fm_ori[:,0])

position = 5
a = 15
b = 14
c = 0
d = 0
# print(improve_count)
# plt.figure(dpi=100)
# plt.plot(estimation[-48*6*5:-48*6*2:,0],color = 'green', label='LSTM')
# plt.plot(estimation_modify[-48*6*5:-48*6*2,0], color = 'red', label='DAW-LSTM')
# plt.legend()
# plt.show()
plt.figure(dpi=150)
plt.plot(r[-48*6*a+c:-48*6*b+d,5],color = 'red', label='real', linewidth = 1)
plt.plot(f[-48*6*a+c:-48*6*b+d,position], color = 'green', label='LSTM forecast', linewidth = 1)
# plt.plot(forecast[-48*6*a+c:-48*6*b+d,position], color = 'black', label='forecast')
# plt.plot(forecast[-48*6*a+c:-48*6*b+d,3], color = 'blue', label='forecast')
plt.plot(dac[-48*6*a+c:-48*6*b+d,position], color = 'blue', label='DAC-LSTM', linewidth = 1)
# plt.plot(f[-48*6*a+c:-48*6*b+d,position], color = 'purple', label='correction with cap')
plt.xlabel('Sample')
plt.ylabel('Power(W)')
plt.legend()
plt.show()

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


# r_av = 1 - ((fm- r) ** 2).sum() / ((fm - r.mean()) ** 2).sum()
# print('m:',r_av)
# r2_av = 1 - ((f - r) ** 2).sum() / ((f- r.mean()) ** 2).sum()
# print(r2_av)
# ma_av1 = np.mean(abs((fm - r) / r))
# print('Test MAPE fm: %.4f' % ma_av1)
# ma_av2 = np.mean(abs((f - r) / r))
# print('Test MAPE: %.4f' % ma_av2)

# k = 1 - ((fm[:,position] - real[:,position]) ** 2).sum() / ((fm[:,position] - real[:,position].mean()) ** 2).sum()
# print('m:',k)
# k2 = 1 - ((forecast[:,position] - real[:,position]) ** 2).sum() / ((forecast[:,position] - real[:,position].mean()) ** 2).sum()
# print(k2)
# ma1 = np.mean(abs((fm[:,position] - real[:,position]) / real[:,position]))
# print('Test MAPE fm: %.3f' % ma1)
# ma2 = np.mean(abs((forecast[:,position] - real[:,position]) / real[:,position]))
# print('Test MAPE: %.3f' % ma2)

# corr, _ = pearsonr(fm[:,5], r[:,1])
# print('Test m Cor: %.3f' % corr)
# corr, _ = pearsonr(f[:,5], r[:,1])
# print('Test Cor: %.3f' % corr)
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
# savecsv4='G:/model_win/TV_general/data/%ihr_%ihousehold_%idays_daclstm_table.csv'%(hr, household,days_for_model)
# aaaaa4.to_csv(savecsv4,index=None)    

    
    
    
    
    
    
    
    