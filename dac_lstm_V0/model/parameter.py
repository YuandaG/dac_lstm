# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 16:02:29 2021

@author: 18636
"""

# Don't set household and GSPs to 0 simoutaniously, because of data missing, script 
# not executable. Details see step1_model.py line 465
household = 220
days_for_model =  550
control = 1 # control = 1 for 5 min data, = 0 for 30 min data
GSPs = 0
long_test = 0 # convert 5min data to 30min data, also execute code with 30min data
predictstep_30min = 2 # 30-min data, how many half-hours want to predict
predictstep_5min = 6 # 5-min data, hour many steps want to predic, 1 step = 5 minutes
pre_time = 24 # using how many hours to predict next step/steps
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

# saving path
path_model_save = os.path.join(path_project,'outputs/model/')
path_model_history = os.path.join(path_project,'outputs/history/')
path_5min = os.path.join(path_project,'outputs/TVVP_5min/')
path_30min = os.path.join(path_project,'outputs/TVVP_30min/')

