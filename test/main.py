import torch
import numpy as np

# Example data normalization
def normalize_data(data):
    return (data - np.mean(data)) / np.std(data)

data = np.arange(0,10,1)
# Assuming `data` is your time series data
data_normalized = normalize_data(data)
data_tensor = torch.FloatTensor(data_normalized).view(-1)

import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq

# Example: Preparing the training sequences with a time window (tw) of 5
time_window = 5
train_inout_seq = create_inout_sequences(data_tensor, time_window)

model = LSTMModel(input_size=1, hidden_layer_size=4, output_size=1)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 100

for i in range(epochs):
    for seq, labels in train_inout_seq:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                             torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if i%25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

model.eval()

test = np.arange(10,30,1)
test_sequences = create_inout_sequences(test, time_window)

with torch.no_grad():
     for seq, labels in train_inout_seq:
        y_pred = model(seq)
        print(labels, y_pred)
