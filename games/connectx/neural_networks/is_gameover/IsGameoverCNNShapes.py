# DOCS: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.ConnectXBBNN import configuration
from neural_networks.is_gameover.BitboardNN import BitboardNN
from neural_networks.is_gameover.device import device


# BUG: 99.8% accuracy may be having trouble with the no_move_moves)
# self.cnn_channels = (10 + 16) | game:  100000 | move: 2130054 | loss: 0.113 | accuracy: 0.854 / 0.953 | time: 587s
# self.cnn_channels = (10 + 16) | game:  200000 | move: 4261355 | loss: 0.092 | accuracy: 0.885 / 0.953 | time: 1156s
# self.cnn_channels = (10 + 16) | game:  300000 | move: 6398089 | loss: 0.082 | accuracy: 0.900 / 0.953 | time: 1724s
# self.cnn_channels = (10 + 16) | game: 1067000 | move: 9928746 | loss: 0.002 | accuracy: 0.998 / 0.953 | time: 4660s
class IsGameoverCNNShapes(BitboardNN):
    def __init__(self):
        super().__init__()
        rows   = configuration.rows
        cols   = configuration.columns
        inarow = configuration.inarow

        self.one_hot_size = 3
        self.input_size   = rows * cols * self.one_hot_size
        self.output_size  = 1
        self.cnn_channels = 4

        # 4x4 *2 for diagonals, 1x4 + 4x1 for horizontal vertical, 6x7 for whole board
        self.convs = [
            nn.Conv2d(self.one_hot_size, self.cnn_channels, (inarow,inarow)),  # 12 out
            nn.Conv2d(self.one_hot_size, self.cnn_channels, (inarow,inarow)),  # 12 out
            nn.Conv2d(self.one_hot_size, self.cnn_channels, (1,inarow)),       # 24 out
            nn.Conv2d(self.one_hot_size, self.cnn_channels, (inarow,1)),       # 21 out
            nn.Conv2d(self.one_hot_size, self.cnn_channels, (rows,cols)),      #  1 out
        ]
        # Small neural network for AND and OR logic
        self.fcs = [
            nn.Linear(70 * self.cnn_channels, 16),
            nn.Linear(16, 4)
        ]
        self.output = nn.Linear(4, self.output_size)

        # Move model weights to CUDA GPU
        self.to(device)
        self.convs = [ conv.to(device) for conv in self.convs ]
        self.fcs   = [ fc.to(device)   for fc   in self.fcs   ]



    # DOCS: https://towardsdatascience.com/understanding-input-and-output-shapes-in-convolution-network-keras-f143923d56ca
    # pytorch requires:    contiguous_format = (batch_size, channels, height, width)
    # tensorflow requires: channels_last     = (batch_size, height, width, channels)
    def cast(self, x):
        x = super().cast(x)                                                           # x.shape = (height, width, channels)
        x = x.view(-1, configuration.rows, configuration.columns, self.one_hot_size)  # x.shape = (batch_size, height, width, channels)
        x = x.permute(0, 3, 1, 2)                                                     # x.shape = (batch_size, channels, height, width)
        return x


    def forward(self, x):
        input = self.cast(x)                # x.shape = (1,3,6,7)
        x_cnns = [
            F.leaky_relu(conv(input))
            for conv in self.convs
        ]
        # BUG: [W TensorIterator.cpp:924] Warning: Mixed memory format inputs detected while calling the operator.
        # The operator will output channels_last tensor even if some of the inputs are not in channels_last format. (function operator())
        x = torch.cat([ x_cnn.permute(0, 3, 1, 2).reshape(x_cnn.shape[0], -1) for x_cnn in x_cnns ], dim=1).view(x.shape[0], -1)

        for fc in self.fcs:
            x = F.leaky_relu(fc(x))
        x = torch.sigmoid(self.output(x))   # x.shape = (-1, 70 * channels, 1)
        x = torch.flatten(x)                # x.shape = (-1, 70 * channels)
        return x


isGameoverCNNShapes = IsGameoverCNNShapes()
isGameoverCNNShapes.to(device)