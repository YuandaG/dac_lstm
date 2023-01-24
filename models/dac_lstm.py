import torch
import torch.nn as nn
import math


class DAC_LSTMModel(nn.Module):

    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.output_size = args.output_size
        self.num_directions = 1 # 2 for bi-direction lstm
        self.batch_size = args.batch_size

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        # for customizing each lstm layer
        # self.lstm1 = nn.LSTM(input_size, hidden_size)
        # self.lstm2 = nn.LSTM(hidden_size, hidden_size)
        # self.lstm3 = nn.LSTM(hidden_size, hidden_size)
        self.linear = nn.Linear(args.hidden_size, args.output_size)
        # self.linear = nn.DAC(args.output_size, args.output_size)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(self.device)
        c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(self.device)
        output, _ = self.lstm(input_seq, (h_0, c_0))
        pred = self.linear(output)
        pred = pred[:, -1, :]

        return pred


class DAC(nn.Module):
    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        weights = torch.Tensor(size_out, size_in)
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
        bias = torch.Tensor(size_out)
        self.bias = nn.Parameter(bias)

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))  # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def forward(self, x):
        w_times_x = torch.mm(x, self.weights.t())
        return torch.add(w_times_x, self.bias)  # w times x + b


class DAC_LSTMModel_CELL(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        # self.args = args
        self.device = device
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.num_directions = 1
        self.batch_size = args.batch_size
        self.lstm0 = nn.LSTMCell(args.input_size, hidden_size=128)
        self.lstm1 = nn.LSTMCell(input_size=128, hidden_size=32)
        self.dropout = nn.Dropout(p=0.4)
        self.linear = nn.Linear(32, self.output_size)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        # batch_size, hidden_size
        h_l0 = torch.zeros(batch_size, 128).to(self.device)
        c_l0 = torch.zeros(batch_size, 128).to(self.device)
        h_l1 = torch.zeros(batch_size, 32).to(self.device)
        c_l1 = torch.zeros(batch_size, 32).to(self.device)
        output = []
        for t in range(seq_len):
            h_l0, c_l0 = self.lstm0(input_seq[:, t, :], (h_l0, c_l0))
            h_l0, c_l0 = self.dropout(h_l0), self.dropout(c_l0)
            h_l1, c_l1 = self.lstm1(h_l0, (h_l1, c_l1))
            h_l1, c_l1 = self.dropout(h_l1), self.dropout(c_l1)
            output.append(h_l1)

        pred = self.linear(output[-1])

        return pred