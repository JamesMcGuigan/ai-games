# The purpose of this neural network is to predict the previous T=-1 state from a game of life board

import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_networks.device import device
from neural_networks.GameOfLifeBase import GameOfLifeBase


class GameOfLifeForward(GameOfLifeBase):
    def __init__(self):
        super().__init__()
        self.one_hot_size     = 1
        self.cnn_kernel_size  = 3  # 3x3 = 1 distance nearest neighbours
        self.dense_layer_size = 9
        #
        # self.input_size       = (25,25)  # manually apply wraparound padding
        # self.output_size      = 25 * 25      # 25 is predefined by kaggle
        # self.cnn_channels     = 2 ** 9       # 512 = every 3x3 pixel combination - may be too much
        # self.cnn_output_size  = self.output_size * self.cnn_channels

        # out_channels=(128,16,4,1)        | epoch: 1936 | board_count: 484000 | loss: 1.01578772 | accuracy = 0.98055 | time: 0.479ms/board
        # out_channels=(128,32,8,1)+1 + l1 | epoch:  416 | board_count: 104000 | loss: 0.15340627 | accuracy = 0.99243 | time: 0.530ms/board
        self.conv1   = nn.Conv2d(in_channels=1,     out_channels=128, kernel_size=(3,3), padding=1, padding_mode='circular')
        self.conv2   = nn.Conv2d(in_channels=128,   out_channels=16,  kernel_size=(1,1))
        self.conv3   = nn.Conv2d(in_channels=16,    out_channels=8,   kernel_size=(1,1))
        self.conv4   = nn.Conv2d(in_channels=1+8,   out_channels=1,   kernel_size=(1,1))
        self.dropout = nn.Dropout(p=0.1)

        # # # epoch: 1936 | board_count: 484000 | loss: 1.01578772 | accuracy = 0.98055 | time: 0.479ms/board
        # self.conv1  = nn.Conv2d(in_channels=1,  out_channels=10,  kernel_size=(3,3), padding=1, padding_mode='circular')
        # self.conv2  = nn.Conv2d(in_channels=10, out_channels=2,   kernel_size=(1,1))
        # self.conv3  = nn.Conv2d(in_channels=2,  out_channels=1,   kernel_size=(1,1))


        # self.fc1    = nn.Linear(self.cnn_output_size,  self.output_size)
        # self.output = nn.Linear(self.output_size,      self.output_size)


    # DOCS: https://towardsdatascience.com/understanding-input-and-output-shapes-in-convolution-network-keras-f143923d56ca
    # pytorch requires:    contiguous_format = (batch_size, channels, height, width)
    # tensorflow requires: channels_last     = (batch_size, height, width, channels)
    def cast_inputs(self, x):
        x = super().cast_inputs(x)   # x.shape = (height, width, channels)
        if len(x.shape) != 4:  #        != (batch_size, channels, height, width)
            x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])   # x.shape = (batch_size, channels, height, width)
        return x

    def forward(self, x):
        x = input = self.cast_inputs(x)

        x = self.conv1(x)
        x = F.leaky_relu(x)

        # x = torch.cat([ x, input ], dim=1)
        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)

        # x = torch.cat([ x, input ], dim=1)
        x = self.conv3(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)

        x = torch.cat([ x, input ], dim=1)
        x = self.conv4(x)
        x = torch.sigmoid(x)

        return x


gameOfLifeForward = GameOfLifeForward()
gameOfLifeForward.to(device)
