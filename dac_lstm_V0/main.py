from model.step1_Model import forecast_main
from model.parameter import *
from pandas import read_csv

if __name__ == "__main__":
    assert os.path.exists(path_project), "Current path invalid, please change path"

    # day = [14,30,50,100,150,200,250,300,350,400,450,500,550]
    # household = [15,30,50,100,150,220]
    household = [220]
    for b in household:
        days_for_model = 365
        household = b
        print('household',household)
        
        # household = 220
        # # days_for_model =  500
        # control = 0
        # GSP = 0

        # print(a)
        
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
        elif household != 0 & GSPs == 0:
            # loaddata = read_csv('G:/LSTM/data_%i.csv'%household, header=0)
            loaddata = read_csv(os.path.join(path_households_data,'data_%i.csv'%household), header=0)
            if long_test == 1:
                loaddata = loaddata['0']
                
            elif long_test == 0:
                loaddata = loaddata['0'].rolling(rolling,min_periods=1).mean()
            # print(loaddata.shape[1])
            m = (loaddata.shape[0]-1)/6+1
            # print(m)
            trans_to_05hr = np.zeros((int(m),1))
            # print('trans shape',trans_to_05hr.shape)
            for i in range(len(trans_to_05hr)):
                trans_to_05hr[i] = loaddata[6*i]
                if trans_to_05hr[i] >120000 or trans_to_05hr[i] <10000:
                    trans_to_05hr[i] = trans_to_05hr[i-1]
            plt.plot(trans_to_05hr[:,0])
            plt.show()
            if long_test == 1:
                loaddata = pd.DataFrame(trans_to_05hr)
                # print('05hr test')
            elif long_test == 0:
                loaddata = pd.DataFrame(loaddata)
                # print('5min test')
            loaddata = loaddata.values
            dataset = np.zeros((loaddata.shape[0],loaddata.shape[1]))
            dataset[:,0] = loaddata[:,0]
            # dataset[:,:] = loaddata[:,:]
            # dataset[:,1] = loaddata[:,1]
            print('TVVP household data loaded')
        elif household == 0 & GSPs == 0:
            # 05hr
            # Data missing
            loaddata = read_csv('G:/LSTM_Prediction/data/weather_data_expanded_05hr_result.csv', header=0)
            dataset = np.zeros((loaddata.shape[0],1))
            loaddata = loaddata.values
            dataset[:,0] = loaddata[:,1]   
        
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
        epoch_earlystop = 20
        scores_plot = []
        
        start_day = 1 #from 1 to 584
        
        
        # how many steps used as input
        # timestep = 24*a*1-predictstep
        timestep = 48*1
        # timestep = 48*6
        drop_times = np.arange(predictstep)
        forecast_main()