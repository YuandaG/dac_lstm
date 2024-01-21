from math import ceil
from math import sqrt
from numpy import concatenate
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping
import pandas as pd
from tensorflow.keras.optimizers import RMSprop
from numba import cuda 
from parameter import *
import operator
import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
import os,sys
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'


def check_path(path):
    if not os.path.exists(path):
    # Create the directory if it doesn't exist
        os.makedirs(path)

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def R2(y_test, y_true):
    return 1 - ((y_test - y_true) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum()

def split_test_train(values):
    # print(n,number_of_days_for_training)
    train = values[((start_day-1)*n):(start_day+number_of_days_for_training-1)*n, :]
    # print(train.shape,1)
    # test = values[(start_day+number_of_days_for_training-1)*48:(start_day+number_of_days_for_training-1)*48+number_of_days_for_prediction*48, :]
    test = values[(start_day+number_of_days_for_training-1)*n:days_for_model*n, :]
    # print(test.shape,1)
    return train, test

def data_split_reshape(train,test,n_feature,n_obs):
    # split into input and outputs
    train_X, train_y = train[:, :n_obs], train[:, -predictstep:]
    test_X, test_y = test[:, :n_obs], test[:, -predictstep:]

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], timestep, n_feature))
    test_X = test_X.reshape((test_X.shape[0], timestep, n_feature))
    # print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    return train_X,train_y,test_X,test_y

# design network
def LSTM_model(n_cells,train_X,train_y,test_X,test_y,epoch_earlystop,n_batch_size):
    model = Sequential()
    # model.add(LSTM(n_cells, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences = True))
    model.add(LSTM(n_cells, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences = False, recurrent_activation = 'sigmoid'))
    # model.add(LSTM(n_cells, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences = False, recurrent_activation = 'sigmoid'))
    # model.add(LSTM(ceil(n_cells/2),return_sequences=True))   
    # model.add(Dropout(0.01))
    # model.add(LSTM(ceil(n_cells/4),return_sequences=True))   
    # model.add(Dropout(0.01))
    # model.add(LSTM(ceil(n_cells/8),return_sequences=False))   
    # model.add(Dropout(0.01))
    model.add(Dense(predictstep))
    model.compile(loss='mse', optimizer=RMSprop(lr=learning_rate), metrics=['mse'])
    # fit network
    es = EarlyStopping(monitor = 'val_loss', patience = epoch_earlystop, restore_best_weights = True, verbose=1)
    history = model.fit(train_X, train_y, epochs=1000
                        
                         , batch_size=n_batch_size, validation_data=(test_X, test_y), verbose=2, shuffle=False, callbacks=[es])
    # history = model.fit(train_X, train_y, epochs=5, batch_size=n_batch_size, validation_data=(test_X, test_y), verbose=0, shuffle=False)
    loss, acc = model.evaluate(test_X, test_y, verbose=0)
    print('Training finished')
    return loss, model, history, acc


def model_prediction(model, test_X, test_y, scaler, n_feature, scaled_values):
    # global scaler
    testprediction = model.predict(test_X)
    # print(testprediction.shape)
    test_X = test_X.reshape((test_X.shape[0], timestep * n_feature))
    inv_testprediction = concatenate((test_X[:, -scaled_values.shape[1] - predictstep:],testprediction), axis=1)
    inv_testprediction = scaler.inverse_transform(inv_testprediction)
    inv_testprediction = inv_testprediction[:, -predictstep:]

    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), predictstep))
    inv_testy = concatenate((test_X[:, -scaled_values.shape[1] - predictstep:],test_y), axis=1)
    inv_testy = scaler.inverse_transform(inv_testy)
    inv_testy = inv_testy[:, -predictstep:]

    return inv_testprediction, inv_testy

# only used for get proper ev data format 
def inv_y(test_X, test_y, scaled_values, scaler):
    test_y = test_y.reshape((len(test_y), predictstep))
    inv_testy = concatenate((test_X[:, -scaled_values.shape[1] - predictstep:],test_y), axis=1)
    inv_testy = scaler.inverse_transform(inv_testy)
    inv_testy = inv_testy[:, -predictstep:]
    return inv_testy

def prepare_data(dataset, timestep, predictstep):
    # extract raw values
    values = dataset.values
    reframed = series_to_supervised(values, timestep, predictstep)
    # drop features we don't want to use for forecast
    drop_columns_count = values.shape[1]
    drop_columns = np.arange(0,0,1)
    for i in drop_times:
        drop_columns = concatenate((drop_columns,np.arange(timestep * drop_columns_count + 1 + drop_columns_count * i, (timestep + 1) * drop_columns_count + drop_columns_count * i, 1)),axis=None)
        # print(drop_columns)
        
    reframed.drop(reframed.columns[drop_columns], axis=1, inplace=True)
    n_feature = drop_columns_count
    n_obs = n_feature * timestep
# 	raw_values = raw_values.reshape(len(raw_values), 1)
	# transform into supervised learning problem X, y
    values = reframed.values
    return values, n_feature, n_obs
# scaler, scaled_values, train, test, n_feature, n_obs
def scale(values):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(values)
    return scaled_values, scaler

def split_train_test(scaled_values):
    # split into train and test sets
    train, test = split_test_train(scaled_values)
    return train, test

def get_ori_testy(loaddata_randomerror):
    
    loaddata_randomerror = pd.DataFrame(loaddata_randomerror)   
    values, n_feature, n_obs = prepare_data(loaddata_randomerror, timestep, predictstep)
    scaled_values, scaler = scale(values)
    train, test = split_train_test(values)
    train_X, train_y, test_X, test_y = data_split_reshape(train, test, n_feature, n_obs)
    test_X = test_X.reshape((test_X.shape[0], timestep * n_feature))
    test_y = test_y.reshape((len(test_y), predictstep))
    inv_testy = concatenate((test_X[:, -scaled_values.shape[1] - predictstep:],test_y), axis=1)
    inv_testy = inv_testy[:, -predictstep:]
    return inv_testy


def forecast_main():
    values, n_feature, n_obs = prepare_data(dataset, timestep, predictstep)
    scaled_values, scaler = scale(values)
    train, test = split_train_test(scaled_values)
    # print(train.shape, test.shape)
    train_X, train_y, test_X, test_y = data_split_reshape(train, test, n_feature, n_obs)

    values_ev, n_feature_ev, n_obs_ev = prepare_data(evdata, timestep, predictstep)
    train_ev, test_ev = split_train_test(values_ev)
    train_X_ev, train_y_ev, test_X_ev, test_y_ev = data_split_reshape(train_ev, test_ev, n_feature_ev, n_obs_ev)

    n_cells = [int(train_X.shape[0]/2/(train_X.shape[1]+train_X.shape[2]))]
    # model training
    for value_cells in n_cells:
        for value_batch_size in n_batch_size:
            #repeat the prediction for n_repeat times
            loss_value = list()
            # yhat_value = list()
            test_X_value = list()
            for i in range(n_repeats):
                # train_X, train_y, test_X, test_y = data_split_reshape(train, test)
                print('start training')
                loss, model, history, acc = LSTM_model(value_cells,train_X,train_y,test_X,test_y,epoch_earlystop,value_batch_size)
                loss_value.append(loss)
                print('>%d/%d cells=%i, batch=%d, loss=%f, lr=%.4f' % (i + 1, n_repeats, value_cells, value_batch_size, loss, learning_rate))
                # test_X_value.append(test_X)
                # scores_plot.append(loss)
            # model.save('C:/Users/s1572867/OneDrive - University of Edinburgh/LSTM_model_PC/model_cell%i_timestep%i_predictstep%i_lr%.3f_batch%i.h5_timeinterval%i_features%i' % (
            #     value_cells, timestep, predictstep, learning_rate, value_batch_size, n, n_feature))  # creates a HDF5 file 'my_model.h5'
    
            # scores[str(value_cells),str(value_batch_size)] = loss_value
            # scores_mean = round(np.mean(scores_plot),4)
            # figure_plot.at[str(value_cells),str(value_batch_size)] = scores_mean
    
    
    ### Notes: for the same column in test_y and testprediction, they represent same forecast result and actual value.
    ### For example, in 5min forecast, if forecast next 3 steps which is 15min forecast, first column in test_y and testprediction
    ### compares the actual value and the next 5min accuracy, the second column represents the actual value and the accuracy
    ### at 10min forecast.
    
    check_path(path_model_save)

    # !mkdir -p saved_model
    model.save(os.path.join(path_model_save,file_name_model))
    inv_testprediction, inv_testy = model_prediction(model, test_X, test_y, scaler, n_feature, scaled_values)
    # inv_testy_ev = inv_y(test_X_ev, test_y_ev, scaled_values_ev, scaler_ev)
    # inv_testprediction_ev, inv_testy_ev = model_prediction(model, test_X_ev, test_y_ev, scaler_ev, n_feature_ev)
    
    
    # plot training result
    # print(scores.describe())
    # box and whisker plot of results
    # scores.boxplot()
    
    
    # # Original test_y
    # loaddata = read_csv('H:/LSTM/load_error_5min.csv', header=0)
    # loaddata_randomerror = loaddata.values[:,0]
    # inv_testy_randomerror = get_ori_testy(loaddata_randomerror)
    
    # Original test_y
    if GSPs == 1:
        loaddata = read_csv(os.path.join(path_GSP_data,'GSP2011_5min_%s.csv'%GSP_index), header=0)
        loaddata_randomerror = -1*loaddata.values[:,0]
        inv_testy_randomerror = get_ori_testy(loaddata_randomerror)
    if household != 0:
        loaddata = read_csv(os.path.join(path_households_data,'data_%i.csv'%household), header=0)
        loaddata_randomerror = loaddata.values[:,0]
        inv_testy_randomerror = get_ori_testy(loaddata_randomerror)
    if future == 1:
        loaddata = read_csv(os.path.join(path_future_data,'future_load.csv'), header=0)
        loaddata_randomerror = loaddata.values[:,3]
        inv_testy_randomerror = get_ori_testy(loaddata_randomerror)
    
    plt_index = predictstep - 1
    
    
    plt.plot(history.history['val_loss'],label = 'validation loss')
    plt.plot(history.history['loss'],label = 'train loss')
    plt.xlabel('epoch')
    plt.ylabel('Validation loss')
    plt.legend()
    plt.show()
    i = np.array(history.history['val_loss'])
    j = np.array(history.history['loss'])
    history_file = np.zeros((i.shape[0],2))
    history_file[:,0] = i
    history_file[:,1] = j
    check_path(path_model_history)
    aaaaa4=pd.DataFrame(history_file)
    savecsv4=os.path.join(path_model_history, file_name_smooth_households_data)
    aaaaa4.to_csv(savecsv4,index=None)
    # plt.figure(figsize=(12,8),dpi=300)
    # plt.plot(inv_testprediction[-48*7:],label = 'prediction')
    # plt.plot(inv_testy[-48*7:],label = 'actual')
    # plt.xticks(np.arange(1,48*7,48),np.arange(1,8,1))
    # plt.xlabel('Days')
    # plt.ylabel('power/W')
    # plt.legend()
    
    
    
    # # 5min plot
    # plt.figure(dpi=500)
    # plt.plot(inv_testy[-48*a*1:,plt_index],label = 'smooth')
    # plt.plot(inv_testy_randomerror[-48*a*1:,plt_index],label = 'Real')
    # plt.plot(inv_testprediction[-48*a*1:,plt_index],label = 'Forecast')
    # plt.legend()
    # plt.xlabel('Sample count');
    # plt.ylabel('Power(W)');
    # plt.show()
    
    # 05hr plot, long time forecast
    plt.figure(dpi=500)
    plt.plot(inv_testy[-48*a*7:,plt_index],label = 'Real')
    plt.plot(inv_testprediction[-48*a*7:,plt_index],label = 'Forecast')
    plt.legend()
    plt.xlabel('Sample count')
    plt.ylabel('Power(W)')
    plt.show()
    
    # # calculate RMSE 5min
    # rmse = sqrt(mean_squared_error(inv_testy_randomerror, inv_testprediction))
    # print('Test RMSE: %.3f' % rmse)
    # R2_value = R2(inv_testprediction[:,plt_index], inv_testy_randomerror[:,plt_index])
    # print('Test R2: %.3f' % R2_value)
    # MAPE = np.mean(abs((inv_testprediction[:,plt_index] - inv_testy_randomerror[:,plt_index]) / (inv_testy_randomerror[:,plt_index]+1)))
    # print('Test MAPE: %.3f' % MAPE)
    
    
    # calculate RMSE 05hr
    rmse = sqrt(mean_squared_error(inv_testy, inv_testprediction))
    print('Test RMSE: %.3f' % rmse)
    R2_value = R2(inv_testprediction[:,plt_index], inv_testy[:,plt_index])
    print('Test R2: %.3f' % R2_value)
    MAPE = np.mean(abs((inv_testprediction[:,plt_index] - inv_testy[:,plt_index]) / (inv_testy[:,plt_index]+1)))
    print('Test MAPE: %.3f' % MAPE)
    print(GSPs, household)
    if GSPs == 1:
        savecsv1='G:/LSTM Data/GSP_result/GSP_5min_%s_smooth.csv'%GSP_index
        savecsv2='G:/LSTM Data/GSP_result/GSP_5min_%s_real.csv'%GSP_index
        savecsv3='G:/LSTM Data/GSP_result/GSP_5min_%s_forecast.csv'%GSP_index
        aaaaa1=pd.DataFrame(inv_testy)
        aaaaa2=pd.DataFrame(inv_testy_randomerror)
        aaaaa3=pd.DataFrame(inv_testprediction)
        aaaaa1.to_csv(savecsv1,index=None)
        aaaaa2.to_csv(savecsv2,index=None)
        aaaaa3.to_csv(savecsv3,index=None)
        print('GSP saved')
    elif household != 0 & GSPs == 4:
        savecsv1='G:/model_win/household comparison/%ihousehold_%i_5min_smooth.csv'%(household,days_for_model)
        savecsv2='G:/model_win/household comparison/%ihousehold_%i_5min_real.csv'%(household,days_for_model)
        savecsv3='G:/model_win/household comparison/%ihousehold_%i_5min_forecast.csv'%(household,days_for_model)
        aaaaa1=pd.DataFrame(inv_testy)
        aaaaa2=pd.DataFrame(inv_testy_randomerror)
        aaaaa3=pd.DataFrame(inv_testprediction)
        aaaaa1.to_csv(savecsv1,index=None)
        aaaaa2.to_csv(savecsv2,index=None)
        aaaaa3.to_csv(savecsv3,index=None)
        print('TVVP Household saved')
    elif (GSPs == 0):
        aaaaa1=pd.DataFrame(inv_testy)
        # aaaaa2=pd.DataFrame(inv_testy_randomerror)
        aaaaa3=pd.DataFrame(inv_testprediction)
        # n=288 -- 5min, n=48 -- 30min
        if n == 48:
            print(f'testing n=48,output path')
            savecsv1=os.path.join(path_30min,file_name_smooth_households_data)
            # savecsv2=os.path.join(path_5min,'%ihr_real_%ihousehold_%idays.csv'%(hr, household,days_for_model))
            savecsv3=os.path.join(path_30min,file_name_forecast_households_data)
            check_path(path_30min)
        # TODO: needs further check, when using 5-min, code below should be executed, instead of above 3 lines of code.
        elif n == 288:
            print(f'testing n=288,output path')
            # savecsv1='G:/model_win/TV_general/%ihr_smooth_%ihousehold_%idays.csv'%(hr, household,days_for_model)
            # savecsv2='G:/model_win/TV_general/%ihr_real.csv'
            # savecsv3='G:/model_win/TV_general/%ihr_forecast_%ihousehold_%idays.csv'%(hr,household,days_for_model)
            savecsv1=os.path.join(path_5min,file_name_smooth_households_data)
            # savecsv2=os.path.join(path_5min,'%ihr_real_%ihousehold_%idays.csv'%(hr, household,days_for_model))
            savecsv3=os.path.join(path_5min,file_name_forecast_households_data)
            check_path(path_5min)
        aaaaa1=pd.DataFrame(inv_testy)
        aaaaa3=pd.DataFrame(inv_testprediction)
        aaaaa1.to_csv(savecsv1,index=None)
        aaaaa3.to_csv(savecsv3,index=None)
        print('TVVP saved')
    # elif (future == 1):
    #     aaaaa1=pd.DataFrame(inv_testy)
    #     # aaaaa2=pd.DataFrame(inv_testy_randomerror)
    #     aaaaa3=pd.DataFrame(inv_testprediction)
    #     if n == 288:
    #         savecsv1='G:/model_win/TV_general/365_5min_smooth.csv'
    #         savecsv2='G:/model_win/TV_general/365_5min_real.csv'
    #         savecsv3='G:/model_win/TV_general/365_5min_forecast.csv'
    #     elif n == 48:
    #         savecsv1='G:/model_win/TV_general/%ihr_smooth_%ihousehold_%idays.csv'%(hr, household,days_for_model)
    #         savecsv2='G:/model_win/TV_general/%ihr_real.csv'
    #         savecsv3='G:/model_win/TV_general/%ihr_forecast_%ihousehold_%idays.csv'%(hr,household,days_for_model)
    #     print('TVVP saved')
    #     aaaaa1=pd.DataFrame(inv_testy)
    #     aaaaa3=pd.DataFrame(inv_testprediction)
    #     aaaaa1.to_csv(savecsv1,index=None)
    #     aaaaa3.to_csv(savecsv3,index=None)
    # Reset GPU and release gpu memory
    device = cuda.get_current_device()
    device.reset()


# Main.py should start from here, needs further work

assert os.path.exists(path_project), "Current path invalid, please change path"

# control = 1 for 5 min data, = 0 for 30 min data
# a: Coefficient, for 5 min a=6, 30 min a=1
# rolling: smooth the curve

if control == 0:
    #05hr
    a = 1
    rolling = 1
    predictstep = predictstep_30min
    timestep = pre_time*a*2-predictstep # default assuming 24hrs = predictstep + timestep
    print("05hr")
elif control == 1:
    #5min
    a = 6
    rolling = 2
    predictstep = predictstep_5min
    timestep = pre_time*a*2-predictstep
    print("5min")

# for b in household:
    # fix the random seed
np.random.seed(5)

data_sample_second = 30*60/a # 30min
# data_sample_second = 30*10 # 5min | 30*60 30min
# n depends on the data sample time interval, 1hr-24, 0.5hr-48, etc.
n = int(24 * 60 * 60 / data_sample_second)


evdata = read_csv(path_ev_data, header=0)

# For GSP
if GSPs == 1:
    household = 0
    loaddata = read_csv(os.path.join(path_GSP_data,'GSP2011_5min_%s.csv'%GSP_index), header=0)
    loaddata = -1*loaddata.rolling(rolling,min_periods=1).mean()
    loaddata=pd.DataFrame(loaddata)
    loaddata = loaddata.values
    dataset = np.zeros((loaddata.shape[0],loaddata.shape[1]))
    dataset[:,0] = loaddata[:,0]
    print('GSP data loaded')
# #For household
elif (household != 0) & (GSPs == 0):
    # loaddata = read_csv(os.path.join(path_households_data,'data_%i.csv'%household), header=0)
    # loaddata = read_csv(os.path.join(path_households_data,'weather_data_expanded_5min_result_loadonly.csv'), header=0) # Test with temp data
    loaddata = read_csv(os.path.join(path_households_data, file_name_households_input_data), header=0) # Test with temp data

    # If need to convert to 30 minutes data
    if long_test == 1:
        # load_data = loaddata['0']
        m = (loaddata.shape[0]-1)/6+1
        try:
            load_data = loaddata['0'].rolling(rolling,min_periods=1).mean()
            trans_to_05hr_load = np.zeros((int(m),1))
        except:
            load_data = loaddata['0'].rolling(rolling,min_periods=1).mean()
            temp_data = loaddata['1'].rolling(rolling,min_periods=1).mean()
            trans_to_05hr_load = np.zeros((int(m),1))
            trans_to_05hr_temp = np.zeros((int(m),1))
        # Select data every 30 minutes and clean data exceed boundary --> usually wrongly recorded data
        for i in range(len(trans_to_05hr_load)):
            trans_to_05hr_load[i] = load_data[6*i]
            try:
                trans_to_05hr_temp[i] = temp_data[6*i]
            except:
                pass
            if trans_to_05hr_load[i] >120000 or trans_to_05hr_load[i] <10000:
                try:
                    trans_to_05hr_load[i] = trans_to_05hr_load[i-1]
                except:
                    pass
        load_data = pd.Series(trans_to_05hr_load.reshape(-1))
        try:
            temp_data = pd.Series(trans_to_05hr_temp.reshape(-1))
        except:
            pass
        # plt.plot(load_data[:])
        # plt.show()

    elif long_test == 0:
        try:
            load_data = loaddata['0'].rolling(rolling,min_periods=1).mean()
        except:
            load_data = loaddata['0'].rolling(rolling,min_periods=1).mean()
            temp_data = loaddata['1'].rolling(rolling,min_periods=1).mean()
    
    for i in range(len(load_data)):
        if load_data[i] >120000 or load_data[i] <10000:
            try:    
                load_data[i] = load_data[i-1]
            except:
                pass
    # Plot the load data t check if there is any obvious errors
    plt.plot(load_data[:])
    plt.show()

    try:    
        loaddata = pd.concat([load_data, temp_data], axis=1)
    except:
        pass

    loaddata = pd.DataFrame(loaddata)
        # print('5min test')
    loaddata = loaddata.values
    # dataset = np.zeros((loaddata.shape[0],loaddata.shape[1]))
    dataset = loaddata
    # dataset[:,:] = loaddata[:,:]
    # dataset[:,1] = loaddata[:,1]
    print('TVVP household data loaded')
elif (household == 0) & (GSPs == 0):
    # 05hr
    # Data missing
    loaddata = read_csv('G:/LSTM_Prediction/data/weather_data_expanded_05hr_result.csv', header=0)
    dataset = np.zeros((loaddata.shape[0],1))
    loaddata = loaddata.values
    dataset[:,0] = loaddata[:,1]
    print(f'wrong data loaded')   

elif future == 1:
    #05hr
    loaddata = read_csv(os.path.join(path_future_data,'future_load.csv'), header=0)
    dataset = np.zeros((loaddata.shape[0],1))
    loaddata = loaddata.values
    dataset[:,0] = loaddata[:,1]    
# loaddata = read_csv('G:/LSTM_Prediction/data/weather_data_expanded_05hr_result.csv', header=0)
# dataset = np.zeros((loaddata.shape[0],1))
# loaddata = loaddata.values
# dataset[:,0] = loaddata[:,1]
    ##5min
    # loaddata = read_csv('G:/LSTM/load_error_5min.csv', header=0)
    # dataset = np.zeros((loaddata.shape[0],loaddata.shape[1]))
    # loaddata = loaddata.values
    # dataset[:,0] = loaddata[:,0]+evdata[:,0]
    # dataset[:,1] = loaddata[:,1]
    print('Future TVVP data loaded')

evdata = evdata.values
# dataset = np.zeros((loaddata.shape[0],loaddata.shape[1]))

#
# dataset[:,0] = loaddata[:,0]
# dataset[:,0] = loaddata[:,0]+evdata[:,0]
# dataset[:,1] = loaddata[:,1]

dataset=pd.DataFrame(dataset)
evdata=pd.DataFrame(evdata)


# defining parameters for LSTM network
learning_rate = 0.001
if days_for_model == 14 or days_for_model == 30 or days_for_model == 50:
    n_batch_size = [48*1]
    number_of_days_for_training = int(2/3*days_for_model)

else:
    n_batch_size = [48*7]
    number_of_days_for_training = int(2/3*days_for_model)



#time of repeat to get average value, when the seed is fixed at the beginning, repeats times can be less
n_repeats = 1
scores = DataFrame()

#time of epoch to stop training while there is no change of loss
epoch_earlystop = 5
scores_plot = []

start_day = 1 #from 1 to 584


# how many steps used as input

# timestep = 48*1
# timestep = 48*6
drop_times = np.arange(predictstep)
print(f'household = {household}\npredictstep = {predictstep}\ntimestep = {timestep}\ndays_for_model = {days_for_model}')
print(f'\nn_batch_size = {n_batch_size}')

forecast_main()