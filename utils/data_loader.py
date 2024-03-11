import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset, DataLoader


def load_data(args):
    df = pd.read_csv(args.file_path, header=None)
    columns = df.columns
    df.fillna(df.mean(), inplace=True)
    return df


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def data_loader(args):
    print('data processing...')
    dataset = load_data(args)
    # split
    # train = dataset[:int(len(dataset) * args.split)]
    # # val = dataset[int(len(dataset) * args.split):int(len(dataset) * 0.8)]
    # test = dataset[int(len(dataset) * args.split):len(dataset)]

    # for mini test
    train = dataset[:int(len(dataset)*3/5)]
    val = dataset[int(len(dataset)*3/5):int(len(dataset)*4/5):]
    test = dataset[int(len(dataset)*4/5):]
    # train = dataset[:1000]
    # test = dataset[1000:2000]

    m, n = np.max(train[train.columns[0]]), np.min(train[train.columns[0]])

    def process(data, batch_size, seq_len, output_size, shuffle):
        load = data[data.columns[0]]
        load = load.tolist()
        data = data.values.tolist()
        # load = (load - n) / (m - n)
        seq = []
        for i in range(len(data) - seq_len - output_size + 1):
            train_seq = []
            train_label = []
            for j in range(i, i + seq_len):
                x = [load[j]]
                train_seq.append(x)
            # for c in range(2, 8):
            #     train_seq.append(data[i + 24][c])\
            for k in range(output_size):
                train_label.append(load[i + seq_len + k])
            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label).view(-1)
            seq.append((train_seq, train_label))

        # print(seq[-1])
        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=True)

        return seq, m, n

    Dtr, _, _ = process(train, args.batch_size, args.seq_len, args.output_size, False)
    # Val = process(val, args.batch_size, True)
    Dva, _, _ = process(test, args.batch_size, args.seq_len, args.output_size, False)
    Dte, m_test, n_test = process(test, args.batch_size, args.seq_len, args.output_size, False)

    print('Done...')
    return Dtr, Dva, Dte, m_test, n_test
