# The purpose of this neural network is to predict the previous T=-1 state from a game of life board

import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_networks.GameOfLifeBase import GameOfLifeBase


class GameOfLifeForward(GameOfLifeBase):
    """
    This implements the life_step() function as a Neural Network function
    Training/tested to 100% accuracy over 10,000 random boards
    """
    def __init__(self):
        super().__init__()

        # epoch: 240961 | board_count: 6024000 | loss: 0.0000000000 | accuracy = 1.0000000000 | time: 0.611ms/board
        # Finished Training: GameOfLifeForward - 240995 epochs in 3569.1s
        self.conv1   = nn.Conv2d(in_channels=1,     out_channels=128, kernel_size=(3,3), padding=1, padding_mode='circular')
        self.conv2   = nn.Conv2d(in_channels=128,   out_channels=16,  kernel_size=(1,1))
        self.conv3   = nn.Conv2d(in_channels=16,    out_channels=8,   kernel_size=(1,1))
        self.conv4   = nn.Conv2d(in_channels=1+8,   out_channels=1,   kernel_size=(1,1))
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = input = self.cast_inputs(x)

        x = self.conv1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)

        x = torch.cat([ x, input ], dim=1)  # remind CNN of the center cell state before making final prediction
        x = self.conv4(x)
        x = torch.sigmoid(x)

        return x


if __name__ == '__main__':
    from neural_networks.train import train

    model = GameOfLifeForward()
    train(model)
