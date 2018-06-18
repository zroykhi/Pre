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

class ConvolutionalNetwork(nn.Module):
    def __init__(self, drop_p=0.2, num_output=2):
        super(ConvolutionalNetwork, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),

            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1)
        )

        self.fc1 = nn.Linear(15088, 128)
        #self.fc1 = nn.Linear(20 * 6 * 26, 16)
        self.fc2 = nn.Linear(128, num_output)
        self.drop_p = drop_p
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

class CNN_RNN(nn.Module):
    def __init__(self, drop_p=0.25, input_size=4032, hidden_size=32, num_layers=2, num_output=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),# False is ok
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),# False is ok
            nn.AvgPool2d(kernel_size=3, stride=2, padding = 1)
        )
        self.drop_p = drop_p
        # self.fc = nn.Linear(15088, input_size)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_output)
        # self.dropout = nn.Dropout(p=0.25)

    def forward(self, inputs, h_state):
        inputs = F.dropout(inputs, p=self.drop_p, training=self.training)
        features = self.cnn(inputs)
        features = features.view(features.size(0), -1)
        features = features[None, :, :]
        # features.backward()
        outputs, h_state = self.rnn(features, h_state)
        # outputs = self.dropout(outputs)
        outs = []    # save all predictions
        for time_step in range(outputs.size(1)):    # calculate output for each time step
            outs.append(self.fc(outputs[:, time_step, :]))
        outs = torch.stack(outs, dim=1)
        outs = outs[0]
        return outs, h_state
