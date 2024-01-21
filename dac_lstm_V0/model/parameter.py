# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 16:02:29 2021

@author: 18636
"""
import csv
# Don't set household and GSPs to 0 simoutaniously, because of data missing, script 
# not executable. Details see step1_model.py line 465
household = 220
days_for_model =  550
control = 1 # control = 1 for 5 min data, = 0 for 30 min data
GSPs = 0
long_test = 0 # convert 5min data to 30min data, also execute code with 30min data
predictstep_30min = 2 # 30-min data, how many half-hours want to predict
predictstep_5min = 6*48 # 5-min data, hour many steps want to predic, 1 step = 5 minutes
pre_time = 24*7 # using how many hours to predict next step/steps
hr = pre_time # avoiding changing all parameters named 'hr' to 'pre_time'
future = 0 

GSP_index = 'A1'

import os
# os.system('all_funs.py')
path_project = './dac_lstm/dac_lstm_V0/'

# data path
path_ev_data = os.path.join(path_project,'data/EV/ev_expand_5min.csv')
path_GSP_data = os.path.join(path_project,'data/GSP/')
path_households_data = os.path.join(path_project,'data/households/')
path_future_data = os.path.join(path_project,'data/TravelPattern/')

# input data
file_name_households_input_data = 'weather_data_expanded_5min_result.csv'
# file_name_households_input_data = 'data_%i.csv'%household
# file_name_households_input_data = 'weather_data_expanded_5min_result_loadonly.csv'


with open(os.path.join(path_households_data, file_name_households_input_data), 'r', newline='', encoding='utf-8') as file:
    # Create a CSV reader object
    reader = csv.reader(file)
    # Read the first row
    first_row = next(reader)
    # Count the number of columns
    feature_number = len(first_row)

# saving data
file_name_model = '%ihousehold_%i_5min_%ifeature.h5'%(household, days_for_model,feature_number)
file_name_smooth_households_data = '%ihr_smooth_%ihousehold_%idays_%ifeature_%ipredstep.csv'%(hr, household,days_for_model,feature_number,predictstep_5min)
file_name_peak_households_data = '%ihr_peak_%ihousehold_%idays_%ifeature_%ipredstep.csv'%(hr, household,days_for_model,feature_number,predictstep_5min)
file_name_forecast_households_data = '%ihr_forecast_%ihousehold_%idays_%ifeature_%ipredstep.csv'%(hr, household,days_for_model,feature_number,predictstep_5min)

# saving path
path_model_save = os.path.join(path_project,'outputs/model/')
path_model_history = os.path.join(path_project,'outputs/history/')
path_5min = os.path.join(path_project,'outputs/TVVP_5min/')
path_30min = os.path.join(path_project,'outputs/TVVP_30min/')

