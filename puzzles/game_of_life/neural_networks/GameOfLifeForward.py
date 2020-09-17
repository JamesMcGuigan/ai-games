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

        self.input_size       = (25+2,25+2)  # manually apply wraparound padding
        self.output_size      = 25 * 25      # 25 is predefined by kaggle
        self.cnn_channels     = 2 ** 9       # 512 = every 3x3 pixel combination - may be too much
        self.cnn_output_size  = self.output_size * self.cnn_channels

        self.conv1  = nn.Conv2d(self.one_hot_size, self.cnn_channels, self.cnn_kernel_size)
        self.fc1    = nn.Linear(self.output_size,  self.output_size)
        self.output = nn.Linear(self.output_size,  self.output_size)


    # DOCS: https://towardsdatascience.com/understanding-input-and-output-shapes-in-convolution-network-keras-f143923d56ca
    # pytorch requires:    contiguous_format = (batch_size, channels, height, width)
    # tensorflow requires: channels_last     = (batch_size, height, width, channels)
    def cast_to_tensor(self, x):
        x = super().cast_to_tensor(x)   # x.shape = (height, width, channels)
        x = x.view(-1, 25, 25, 1)       # x.shape = (batch_size, height, width, channels)
        x = x.permute(0, 3, 1, 2)       # x.shape = (batch_size, channels, height, width)
        return x


    def forward(self, x):
        x = self.cast_to_tensor(x)
        x = F.leaky_relu(self.conv1(x))
        x = x.permute(0,2,3,1).reshape(x.shape[0], -1)  # convert to columns_last (batch_size, height, width, channels)
        x = F.leaky_relu(self.fc1(x))
        x = torch.sigmoid(self.output(x))
        x = x.flatten()
        return x


gameOfLifeForward = GameOfLifeForward()
gameOfLifeForward.to(device)
