"""
Different neural network architectures for detecting the line
"""
from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from .constants import INPUT_HEIGHT, INPUT_WIDTH
class MlpNetwork(nn.Module):
    """
    MLP network for detecting the line
    :param input_dim: (int) 3 x H x W
    :param n_hidden: [int]
    :param drop_p: (float) Dropout proba
    """

    def __init__(self, input_dim, n_hidden=None, drop_p=0.0):
        super(MlpNetwork, self).__init__()
        if n_hidden is None:
            n_layer1 = 20
            n_layer2 = 4
        else:
            n_layer1, n_layer2 = n_hidden
        self.fc1 = nn.Linear(input_dim, n_layer1)
        self.fc2 = nn.Linear(n_layer1, n_layer2)
        self.fc3 = nn.Linear(n_layer2, 1)
        self.drop_p = drop_p
        self.activation_fn = F.relu
        self._initializeWeights()

    def _initializeWeights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                pass
                # m.weight.data.normal_(0, 0.005)
                # m.weight.data.uniform_(-0.005, 0.005)
                # nn.init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                # nn.init.kaiming_normal(m.weight.data)
                # nn.init.kaiming_uniform(m.weight.data)
                # m.bias.data.zero_()
                # m.bias.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_fn(self.fc2(x))
        x = self.activation_fn(self.fc3(x))
        return x

    def customForward(self, x):
        """
        Return intermediate results
        """
        x = x.view(x.size(0), -1)
        x1 = self.activation_fn(self.fc1(x))
        x2 = self.activation_fn(self.fc2(x1))
        x = self.activation_fn(self.fc3(x2))
        return x, x1, x2


# class ConvolutionalNetwork(nn.Module):
#     def __init__(self, drop_p=0.00, num_output=4):
#         super(ConvolutionalNetwork, self).__init__()
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(3, 20, kernel_size=7, stride=1, padding=3),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
#
#             nn.Conv2d(20, 20, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2)
#         )
#
#         # Factor = 1
#         # self.fc1 = nn.Linear(20 * 24 * 106, 16)
#         # Factor = 2
#         # self.fc1 = nn.Linear(20 * 12 * 53, 16)
#         # Factor = 4
#         #self.fc1 = nn.Linear(20 * 6 * 26, 16)
#         self.fc1 = nn.Linear(36260, 64)
#         self.fc2 = nn.Linear(64, num_output)
#         self.drop_p = drop_p
#
#     def forward(self, x):
#         # print(x.shape) # [4, 3, 75, 100]
#         x = F.dropout(x, p=self.drop_p, training=self.training)
#         x = self.conv_layers(x)
#         # print(x.shape) # [4, 20, 37, 49]
#         x = x.view(x.size(0), -1)
#         # print(x.shape) # [4, 36260]
#         x = F.relu(self.fc1(x))
#         # print(x.shape) # [4, 64]
#         x = self.fc2(x)
#         # print(x.shape) # [4, 4]
#
#         return x

# modified by zhi
class ConvolutionalNetwork(nn.Module):
    def __init__(self, drop_p=0.2, num_output=3):
        super(ConvolutionalNetwork, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=5, stride=1, padding=2),

            nn.Conv2d(20, 40, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(40),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1)
        )

        self.fc1 = nn.Linear(37720, 64)
        #self.fc1 = nn.Linear(20 * 6 * 26, 16)
        self.fc2 = nn.Linear(64, num_output)
        self.drop_p = drop_p
        # modified by zhi
        # self.bn1 = nn.BatchNorm2d(3)
        # self.bn2 = nn.BatchNorm2d(24)

    def forward(self, x):
        # x = self.bn1(x)
        # print(type(x))
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.conv_layers(x)
        # print(x.shape)
        # x = self.bn2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # print(type(x))
        # print(x.shape)
        return x

Convo1 = torch.nn.Sequential(
        torch.nn.Conv2d(3, 20, kernel_size=7, stride=1, padding=3),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1),

        torch.nn.Conv2d(20, 20, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=3, stride=2),
        # torch.nn.Linear(36260, 64),
        # torch.nn.Linear(64, 4)
    )
# class Convo(nn.Module):
#     def __init__(self, drop_p=0.0):
#         super(Convo, self).__init__()
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(3, 20, kernel_size=7, stride=1, padding=3),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
#
#             nn.Conv2d(20, 20, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2)
#         )
#         self.fc1 = nn.Linear(36260, 64)
#         self.drop_p = drop_p
#
#     def forward(self, x):
#         x = F.dropout(x, p=self.drop_p, training=self.training)
#         x = self.conv_layers(x)
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#
# cnn = Convo()
# # cnn = Convo1
# pretrained_dict = torch.load("cnn_model.pth")
# cnn_dict = cnn.state_dict()
# # 1. filter out unnecessary keys
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in cnn_dict}
# # 2. overwrite entries in the existing state dict
# cnn_dict.update(pretrained_dict)
# # 3. load the new state dict
# cnn.load_state_dict(cnn_dict)
#
# class cnn_rnn(nn.Module):
#     def __init__(self, cnn, num_ftrs, rnn_hidden, rnn_layers):
#         super().__init__()
#         self.cnn = cnn
#         self.rnn = nn.LSTM(num_ftrs, rnn_hidden, rnn_layers)
#         self.fc = nn.Linear(rnn_hidden, 4)
#         self.fc2 = nn.Linear(80, 4)
#
#     def forward(self, inputs):
#         # features = []
#         # for i, inp in enumerate(inputs):
#         #     print(inp.shape) # [3, 75, 100]
#         #     print(inputs.shape) # [4, 3, 75, 100]
#         #     outs = self.cnn.forward(inp)
#         # # outs = self.cnn.forward(inputs)
#         #     # outs = self.cnn.conv_layers(inp)
#         # # outs = self.cnn.conv_layers(inputs)
#         # # outs = outs.view(outs.size(0), -1)
#         # # outs = outs.view(outs.size(0), -1)
#         #     features.append(outs)
#         # print(inputs.shape) # [4, 3, 75, 100]
#         # print(inputs)
#         # features = self.cnn.forward(inputs)
#         features = self.cnn.conv_layers(inputs)
#         # print(features)
#         # print(features.shape) # [4, 20, 37, 49]
#         # print(type(features))
#         features = features.view(features.size(0), features.size(1), -1)
#         # print(features.shape) # [4, 20, 1813]
#         # features = torch.cat(features).view(64, -1, 64)
#         # features = Variable(features)
#         # features = np.array(outs)
#         # features = torch.from_numpy(features).float()
#         # features = Variable(features)
#         # format features and store in Variable
#         # features = torch.stack(features)
#         # pass through LSTM
#         outputs, _ = self.rnn(features)
#         outputs = outputs.view(outputs.size(0), -1)
#         outputs = F.relu(outputs)
#         # print(outputs.shape) # [4, 20, 4]
#         # outputs = self.fc(outputs[-1])
#         outputs = self.fc2(outputs)
#         # print(outputs.shape) # [20, 4]
#         return outputs
#
# newModel = cnn_rnn(cnn,1813,4,4)
# print(newModel)

# class RNN(nn.Module):
#     def __init__(self):
#     # def __init__(self, time_step, batch_size):
#         super(RNN, self).__init__()
#
#         self.rnn = nn.RNN(
#             input_size=3*162*90,
#             hidden_size=32,     # rnn hidden unit
#             num_layers=20,       # number of rnn layer
#             batch_first=True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
#         )
#         self.out = nn.Linear(32, 3)
#         # self.time_step = time_step
#         # self.batch_size = batch_size
#
#     def forward(self, x, h_state):
#         # turn into (batch, 3*time_step*input_size)
#         x = x.view(x.size(0), -1)
#         # (1, batch, 3*time_step*input_size): a made (batch,time_step, input_size)
#         x = x[None, :, :]
#         # x (batch, time_step, input_size)
#         # h_state (n_layers, batch, hidden_size)
#         # r_out (batch, time_step, hidden_size)
#         r_out, h_state = self.rnn(x, h_state)
#         # print("r_out is", r_out.shape)
#         outs = []    # save all predictions
#         for time_step in range(r_out.size(1)):    # calculate output for each time step
#             outs.append(self.out(r_out[:, time_step, :]))
#         # print("outs len is: ", len(outs))
#         # print("outs element len is: ", len(outs[0][0]))
#         outs = torch.stack(outs, dim=1)
#         # outs = outs.view(self.time_step*self.batch_size,3)
#         outs = outs.view(4, 3)
#         # print("after view: ", outs.shape)
#         return outs, h_state

# when you use rnn model, train batch size must equal to test's
class RNN(nn.Module):
    # here, time step is batch size
    # batch size: 3*width*height (photo)
    def __init__(self, time_step, batch_size):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=3*INPUT_WIDTH*INPUT_HEIGHT,
            hidden_size=128,     # rnn hidden unit
            num_layers=128,      # number of rnn layer
            batch_first=True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(128, 3)
        self.time_step = time_step
        self.batch_size = batch_size

    def forward(self, x, h_state):
        # turn into (batch, 3*time_step*input_size)
        x = x.view(x.size(0), -1)
        # (1, batch, 3*time_step*input_size): a made (batch,time_step, input_size)
        x = x[None, :, :]
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        r_out, h_state = self.rnn(x, h_state)
        # print("r_out is", r_out.shape)
        outs = []    # save all predictions
        for time_step in range(r_out.size(1)):    # calculate output for each time step
            outs.append(self.out(r_out[:, time_step, :]))
        # print("outs len is: ", len(outs))
        # print("outs element len is: ", len(outs[0][0]))
        outs = torch.stack(outs, dim=1)
        # print(outs.shape)
        # outs = outs.view(self.time_step,3)
        outs = outs[0]
        # print("after view: ", outs.shape)
        return outs, h_state
