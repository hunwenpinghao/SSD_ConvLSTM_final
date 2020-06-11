import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = int((kernel_size - 1) / 2)
        # add attention
        self.att_12 = nn.Conv2d(self.input_channels+self.hidden_channels, self.input_channels+self.hidden_channels,
                                self.kernel_size, 1, self.padding, bias=False)
        self.att_3 = nn.Conv2d(self.input_channels + self.hidden_channels, 1, self.kernel_size, 1,
                                self.padding, bias=False)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None # 窥探层
        self.Wcf = None
        self.Wco = None

    def attention(self, x, h):
        # add attention
        att = self.att_12(torch.cat([x, h], 1))
        att = F.relu(att, inplace=True)
        att = self.att_12(att)
        att = F.relu(att, inplace=True)
        alpha = torch.sigmoid(self.att_3(att))
        # feature = alpha.data.cpu().numpy()[0][0]
        # plt.matshow(feature)
        # plt.show()
        x_att = alpha * x
        return x_att, alpha

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
            self.Wcf = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
            self.Wco = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda(), # [1,512,h,w]
                Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda())


class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    # hidden_channels 相当于卷积核的个数
    def __init__(self, input_channels, hidden_channels, kernel_size, step=5, effective_step=[4]):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
            setattr(self, name, cell)
            self._all_layers.append(cell)

        self.out_channels = hidden_channels[-1]


    # def forward(self, input, Attention=True):
    #     self.internal_state = []
    #     for step in range(self.step):
    #         x = input
    #         for i in range(self.num_layers): # 这里层数对应cell的个数，也即是卷积层的层数,各层之间串联
    #             # all cells are initialized in the first step
    #             name = 'cell{}'.format(i)
    #             if step == 0:
    #                 bsize, _, height, width = x.size() # bsize 是序列的个数，即同时训练bsize个序列
    #                 (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
    #                                                          shape=(height, width))
    #                 self.internal_state.append((h, c))
    #
    #             # do forward
    #             (h, c) = self.internal_state[i]
    #             if self.num_layers == 1: # add attention only at the first layer
    #                 x, alpha = getattr(self, name).attention(x, h)
    #             elif i == 0:
    #                 x, alpha = getattr(self, name).attention(x, h)
    #
    #             x, c = getattr(self, name)(x, h, c) # 运行每一个cell的前向推断，得到新的 ht,new_c
    #             self.internal_state[i] = (x, c) # 每一层的状态更新
    #
    #         # only record effective steps
    #         if step in self.effective_step:
    #             outputs = x
    #
    #     return outputs


    def forward(self, input, Attention=True):
        bz, ch, h, w = input.size()
        self.internal_state = []
        outputs = []
        for batch in range(bz):
            for step in range(self.step):
                datain = input[batch]  # [1,ch,h,w]
                datain = torch.unsqueeze(datain, 0)
                for i in range(self.num_layers): # 这里层数对应cell的个数，也即是卷积层的层数,各层之间串联
                    # all cells are initialized in the first step
                    name = 'cell{}'.format(i)
                    if step == 0:
                        bsize, _, height, width = datain.size() # bsize 是序列的个数，即同时训练bsize个序列
                        (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                                 shape=(height, width))
                        self.internal_state.append((h, c))
                    else:
                        # do forward
                        (h, c) = self.internal_state[i]

                        if self.num_layers == 1: # add attention only at the first layer
                            x, alpha = getattr(self, name).attention(datain, h)
                        elif i == 0:
                            x, alpha = getattr(self, name).attention(datain, h)

                        x, c = getattr(self, name)(x, h, c) # 运行每一个cell的前向推断，得到新的 ht,new_c
                        self.internal_state[i] = (x, c) # 每一层的状态更新

                # only record effective steps
                if step in self.effective_step:
                    if batch == 0:
                        outputs = x
                    else:
                        outputs = torch.cat([outputs,x], 0)

        return outputs


