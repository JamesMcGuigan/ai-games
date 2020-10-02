# The purpose of this neural network is to predict the previous T=-1 state from a game of life board
from typing import TypeVar

import torch
import torch.nn as nn

from neural_networks.GameOfLifeBase import GameOfLifeBase

# noinspection PyTypeChecker
T = TypeVar('T', bound='GameOfLifeHardcoded')
class GameOfLifeHardcoded(GameOfLifeBase):
    """
    This implements the life_step() function as a minimalist Neural Network function with hardcoded weights
    Subclasses implement the effect of different activation functions and weights
    """
    def __init__(self):
        super().__init__()

        self.identity   = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), bias=False)  # We need at least one trainable layer
        self.counter    = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(3,3),
                                    padding=1, padding_mode='circular', bias=False)
        self.logics     = nn.ModuleList([
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(1,1))
        ])
        self.output     = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(1,1))
        self.activation = nn.Identity()


    def forward(self, x):
        x = input = self.cast_inputs(x)

        x = self.identity(x)  # noop - a single node linear layer - torch needs at least one trainable layer
        x = self.counter(x)   # counter counts above 6, so no ReLU6

        for logic in self.logics:
            x = logic(x)
            x = self.activation(x)

        x = self.output(x)
        x = torch.sigmoid(x)

        return x



    # Load / Save Functionality

    def load_state_dict(self, **kwargs):
        return self

    def save(self):
        return self

    def unfreeze(self: T) -> T:
        super().unfreeze()
        self.freeze()
        for name, parameter in self.named_parameters():
            if name.split('.')[0] in [ 'identity' ]:
                parameter.requires_grad = True
        return self







