# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 16:02:29 2021

@author: 18636
"""

# Don't set household and GSPs to 0 simoutaniously, because of data missing, script 
# not executable. Details see step1_model.py line 465
household = 220
days_for_model =  550
control = 1
GSPs = 0
long_test = 0
hr = 24
future = 1
GSP_index = 'A1'

import os
# os.system('all_funs.py')
path_project = './dac-lstm/dac_lstm_V0/'

# data path
path_ev_data = os.path.join(path_project,'data/EV/ev_expand_5min.csv')
path_GSP_data = os.path.join(path_project,'data/GSP/')
path_households_data = os.path.join(path_project,'data/households/')
path_future_data = os.path.join(path_project,'data/TravelPattern/')

# saving path
path_model_save = os.path.join(path_project,'outputs/model/')
path_model_history = os.path.join(path_project,'outputs/history/')



