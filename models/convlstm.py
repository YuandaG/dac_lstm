from multiprocessing.sharedctypes import Value
import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        #input_dim是每个num_layer的第一个时刻的的输入dim，即channel
        #hidden_dim是每一个num_layer的隐藏层单元，如第一层是64，第二层是128，第三层是128
        #kernel_size是卷积核
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        #padding的目的是保持卷积之后大小不变
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,#卷积输入的尺寸
                              out_channels=4 * self.hidden_dim,#因为lstmcell有四个门，隐藏层单元是rnn的四倍
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        #input_tensor的尺寸为（batch_size，channel，weight，width），没有time_step
        #cur_state的尺寸是（batch_size,（hidden_dim）channel，weight，width），是调用函数init_hidden返回的细胞状态

        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        #conv层的卷积不需要和linear一样，可以是多维的，只要channel数目相同即可

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        #使用split函数把输出4*hidden_dim分割成四个门
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g   #下一个细胞状态
        h_next = o * torch.tanh(c_next)  #下一个hc

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))



class ConvLSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)
        #核对尺寸，用的函数是静态方法

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        #kernel_size==hidden_dim=num_layer的维度，因为要遍历num_layer次
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        #如果return_all_layers==true，则返回所有得到h，如果为false，则返回最后一层的最后一个h

        cell_list = []
        for i in range(0, self.num_layers):
            #判断input_dim是否是第一层的第一个输入，如果是的话则使用input_dim，否则取第i层的最后一个hidden_dim的channel数作为输入
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        #以num_layer为三层为例，则cell_list列表里的内容为[convlstmcell0（），convlstmcell1（），convlstmcell2（）]
        #Module_list把nn.module的方法作为列表存放进去，在forward的时候可以调用Module_list的东西，cell_list【0】，cell_list【1】，
        #一直到cell_list【num_layer】，
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        #第一次传入hidden_state为none
        #input_tensor的size为（batch_size,time_step,channel,height,width）
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        #在forward里开始构建模型，首先把input_tensor的维度调整，然后初始化隐藏状态
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            #调用convlstm的init_hidden方法不是lstmcell的方法
            #返回的hidden_state有num_layer个hc，cc
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)#取time_step
        cur_layer_input = input_tensor

        #初始化h之后开始前向传播
        for layer_idx in range(self.num_layers):
            #在已经初始化好了的hidden_state中取出第num_layer个状态给num_layer的h0，c0，其作为第一个输入
            h, c = hidden_state[layer_idx]
            output_inner = []
            #开始每一层的时间步传播
            for t in range(seq_len):
                #用cell_list[i]表示第i层的convlstmcell，计算每个time_step的h和c
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                #将每一次的h存放在output_inner里
                output_inner.append(h)
            #layer_output是五维向量，在dim=1的维度堆栈，和input_tensor的维度保持一致
            layer_output = torch.stack(output_inner, dim=1)
            #吧每一层输出肚饿五维向量作为下一层的输入，因为五维向量的输入没有num_layer，所以每一层的输入都要喂入五维向量
            cur_layer_input = layer_output
            #layer_output_list存放的是第一层，第二层，第三层的每一层的五维向量，这些五维向量作为input_tensor的输入
            layer_output_list.append(layer_output)
            #last_state_list里面存放的是第一层，第二层，第三次最后time_step的h和c
            last_state_list.append([h, c])

        if not self.return_all_layers:
            #如果return_all_layers==false的话，则返回每一层最后的状态，返回最后一层的五维向量，返回最后一层的h和c
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]


        return layer_output_list, last_state_list


def _init_hidden(self, batch_size, image_size):
    init_states = []
    for i in range(self.num_layers):
        init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
    return init_states


@staticmethod
def _check_kernal_size_consistency(kernel_size):
    if not (isinstance(kernal_size, tuple) or \
        (isinstance(kernal_size, list) and all([instance(elem, tuple) for elem in kernel_size])))

        raise ValueError('`kernel_size` must be tupler or list of tuples')

@staticmethod
def _extend_for_multilayer(param, num_layers):
    if not isinstance(param, list):
        param = [param] * num_layers
    return param