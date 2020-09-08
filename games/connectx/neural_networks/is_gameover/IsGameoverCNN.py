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
class IsGameoverCNN(BitboardNN):
    def __init__(self):
        super().__init__()
        self.one_hot_size     = 3
        self.input_size       = configuration.rows * configuration.columns * self.one_hot_size
        self.output_size      = 1
        self.cnn_channels     = (10 + 16)  # 4 vertical, 4 horizontal, 2 diagonal lines = 10 + 16 squares
        self.cnn_kernel_size  = configuration.inarow
        self.cnn_output_size  = (configuration.rows-self.cnn_kernel_size+1) * (configuration.columns-self.cnn_kernel_size+1) * self.cnn_channels
        self.dense_layer_size = self.cnn_output_size // 2

        self.conv1  = nn.Conv2d(self.one_hot_size, self.cnn_channels, self.cnn_kernel_size)
        # AdaptiveMaxPool2d(1) == GlobalMaxPool reduces the board to a single pixel per cnn_channel
        # NOTE: this cannot compute is_gameover() as we need detection for no move moves left
        # self.pool   = nn.AdaptiveMaxPool2d(1)
        self.fc1    = nn.Linear(self.cnn_output_size,    self.cnn_output_size//2)
        self.fc2    = nn.Linear(self.cnn_output_size//2, self.cnn_output_size//4)
        self.fc3    = nn.Linear(self.cnn_output_size//4, self.cnn_output_size//8)
        self.output = nn.Linear(self.cnn_output_size//8, self.output_size)


    # DOCS: https://towardsdatascience.com/understanding-input-and-output-shapes-in-convolution-network-keras-f143923d56ca
    # pytorch requires:    contiguous_format = (batch_size, channels, height, width)
    # tensorflow requires: channels_last     = (batch_size, height, width, channels)
    def cast(self, x):
        x = super(IsGameoverCNN, self).cast(x)                                        # x.shape = (height, width, channels)
        x = x.view(-1, configuration.rows, configuration.columns, self.one_hot_size)  # x.shape = (batch_size, height, width, channels)
        x = x.permute(0, 3, 1, 2)                                                     # x.shape = (batch_size, channels, height, width)
        return x


    def forward(self, x):
        x = self.cast(x)                    # x.shape = (1,3,6,7)
        x = F.relu(self.conv1(x))           # x.shape = (1,26,3,4)
        # x = self.pool(x)                  # x.shape = (1,26,1,1)  - this will break is_gameover() logic
        # NOTE: reshape(-1) required for transition into dense layers
        # WARNING (ignore):  Mixed memory format inputs detected - https://github.com/pytorch/pytorch/issues/42300
        x = x.permute(0,2,3,1).reshape(x.shape[0], -1)  # x.shape = (-1, 312, 1) + convert to columns_last (batch_size, height, width, channels)
        x = F.relu(self.fc1(x))             # x.shape = (-1, 156, 1)
        x = F.relu(self.fc2(x))             # x.shape = (-1, 78, 1)
        x = F.relu(self.fc3(x))             # x.shape = (-1, 39, 1)
        x = torch.sigmoid(self.output(x))   # x.shape = (-1, 1, 1)
        x = x.flatten()                     # x.shape = (-1, 1)
        return x


isGameoverCNN = IsGameoverCNN()
isGameoverCNN.to(device)