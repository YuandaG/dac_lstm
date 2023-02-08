import time
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
import gc
import warnings
import matplotlib.pyplot as plt
import datetime
import json

# from config import MODEL_PATH  # HP
from models.dac_lstm import DAC_LSTMModel   # model
from utils.data_loader import data_loader   # dataset
from utils.strategy import set_seed, R2, MAPE  # strategy
from config.config_daclstm import parse_args
from train import train_full
from test import test_full

gc.collect()                        # clear cache
torch.cuda.empty_cache()            # clear memory
warnings.filterwarnings('ignore')   # ignore some warnings


def main():
    set_seed(42)
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load data
    train, test, m_test, n_test = data_loader(args)

    # model
    model = DAC_LSTMModel(args, device).to(device)
    loss_fn = nn.MSELoss().to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    lambda1 = lambda epoch: epoch // 3
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    print(model)

    # month, day, hour, minute,
    thetime = datetime.datetime.now().strftime('%m_%d_%H_%M')
    folder_name = thetime+'_'+os.path.splitext(os.path.split(args.file_path)[-1])[0]+'_'+str(args.input_size)+'_'+str(args.seq_len)+'_'+str(args.output_size)

    args.checkpoint = os.path.join(args.checkpoint, folder_name + '/')

    '''''''''''''''''''''''''''''''''Train'''''''''''''''''''''''''''''''''
    if args.mode == '0' or args.mode == '2':
        # create workplace
        if not os.path.exists(args.checkpoint):
            os.mkdir(args.checkpoint)
        # check if the loss file path exists
        loss_path = os.path.join(args.checkpoint, 'train_loss/')
        if not os.path.exists(loss_path):
            os.mkdir(loss_path)

        start_time = time.time()

        # train
        train_loss_values, model_path = train_full(model, optimizer, loss_fn, train, device, args, scheduler)

        end_time = time.time()
        time_cost = end_time - start_time
        print(f'training cost time == {time_cost}s')
        print(f'loss file saved to {loss_path}')

        # save training losses
        loss_data = json.dumps(train_loss_values)
        loss_file = open(loss_path+"train_loss.json", "w")
        loss_file.write(loss_data)
        loss_file.close()

        # Plot the loss curves
        plt.plot(train_loss_values, label="Train loss")
        plt.title("Training and test loss curves")
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.legend()
        plt.show(block=True)

    '''''''''''''''''''''''''''''''''Test'''''''''''''''''''''''''''''''''
    if args.mode == '1' or args.mode == '2':
        # inherit the best model from training in mode 2.
        if args.mode == '2':
            args.load_ckpt = os.path.join('./', model_path)
            print('*******************Start testing*******************')

        y, pred = test_full(model, args, test, device, m_test, n_test)

        print('mape:', MAPE(y[:,args.output_size-1], pred[:,args.output_size-1]))
        print('R2:', R2(y[:,args.output_size-1], pred[:,args.output_size-1]))
        plt.figure(dpi=150)

        plt.plot(y[0:6*48, args.output_size-1], label='Real')
        plt.plot(pred[0:6*48, args.output_size-1], label='Forecast')
        plt.legend()
        plt.xlabel('Sample count')
        plt.ylabel('Power(W)')
        plt.show(block=True)


if __name__ == '__main__':
    main()