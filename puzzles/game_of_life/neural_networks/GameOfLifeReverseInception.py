from abc import ABCMeta
from typing import TypeVar

import numpy as np
import torch
import torch as pt
import torch.nn.functional as F
from torch import nn

from neural_networks.GameOfLifeBase import GameOfLifeBase
from neural_networks.hardcoded.GameOfLifeHardcodedReLU1_21 import GameOfLifeHardcodedReLU1_21
from neural_networks.modules.ReLUX import ReLU1


class Counter2D(nn.Module):
    """ Count the number of neighbouring cells (including self) """
    def __init__(self, channels=1):
        super().__init__()
        self.counter = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3,3),
                                 padding=1, padding_mode='circular', bias=False)
        self.counter.weight.data = torch.tensor([
            [[[ 1.0, 1.0, 1.0 ],
              [ 1.0, 1.0, 1.0 ],
              [ 1.0, 1.0, 1.0 ]]]
        ])
        self.counter.weight.requires_grad = False

    def forward(self, x):
        return self.counter(x)


class CountNeighbours2D(nn.Module):
    """ Count the number of neighbouring cells (excluding self)"""
    def __init__(self, channels=1):
        super().__init__()
        self.counter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                 kernel_size=(3,3), padding=1, padding_mode='circular', bias=False)
        self.counter.weight.data = torch.tensor([
            [[[ 1.0, 1.0, 1.0 ],
              [ 1.0, 0.0, 1.0 ],
              [ 1.0, 1.0, 1.0 ]]]
        ])
        self.counter.weight.requires_grad = False

    def forward(self, x):
        return self.counter(x)


# noinspection PyTypeChecker
T = TypeVar('T', bound='GameOfLifeFeatures')
class GameOfLifeFeatures(nn.Module):
    """ Count the number of neighbouring cells (excluding self)"""
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Identity(),
            GameOfLifeHardcodedReLU1_21(),
            Counter2D(),
            CountNeighbours2D(),
        ])
        for name, parameter in self.named_parameters(): parameter.requires_grad = False

    def forward(self, x):
        output = torch.cat([
            layer(x[:,n:n+1,:,:])
            for n in range(x.shape[1])
            for layer in self.layers
        ], dim=1)
        return output




class GameOfLifeReverseInception(GameOfLifeBase, metaclass=ABCMeta):
    """
    """
    def __init__(self):
        super().__init__()

        # self.criterion = FocalLoss()
        # self.criterion = nn.BCELoss()
        self.criterion  = nn.MSELoss()
        self.activation = nn.PReLU()
        self.relu1      = ReLU1()

        # discriminator must be at top-level for autograd to work, its weights are added to the savefile
        self.discriminator = GameOfLifeHardcodedReLU1_21()
        for name, parameter in self.discriminator.named_parameters(): parameter.requires_grad = False

        self.features = GameOfLifeFeatures()

        self.layers = nn.ModuleList([

            # self.features,
            nn.ModuleList([
                nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(1,1), padding=0, padding_mode='circular'),
                nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(3,3), padding=1, padding_mode='circular'),
                nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(5,5), padding=2, padding_mode='circular'),
                nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(7,7), padding=3, padding_mode='circular'),
                nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(9,9), padding=4, padding_mode='circular'),
            ]),
            nn.Conv2d(in_channels=32*5, out_channels=32, kernel_size=(1,1), padding=0, padding_mode='circular'),
            nn.Conv2d(in_channels=32,   out_channels=16, kernel_size=(1,1), padding=0, padding_mode='circular'),

            # self.features,
            nn.ModuleList([
                nn.Conv2d(in_channels=16*4+4, out_channels=32, kernel_size=(1,1), padding=0, padding_mode='circular'),
                nn.Conv2d(in_channels=16*4+4, out_channels=32, kernel_size=(3,3), padding=1, padding_mode='circular'),
                nn.Conv2d(in_channels=16*4+4, out_channels=32, kernel_size=(5,5), padding=2, padding_mode='circular'),
                nn.Conv2d(in_channels=16*4+4, out_channels=32, kernel_size=(7,7), padding=3, padding_mode='circular'),
                nn.Conv2d(in_channels=16*4+4, out_channels=32, kernel_size=(9,9), padding=4, padding_mode='circular'),
            ]),
            nn.Conv2d(in_channels=32*5, out_channels=32, kernel_size=(1,1), padding=0, padding_mode='circular'),
            nn.Conv2d(in_channels=32,   out_channels=16, kernel_size=(1,1), padding=0, padding_mode='circular'),

            nn.ModuleList([
                nn.Conv2d(in_channels=16*4+4, out_channels=32, kernel_size=(1,1), padding=0, padding_mode='circular'),
                nn.Conv2d(in_channels=16*4+4, out_channels=32, kernel_size=(3,3), padding=1, padding_mode='circular'),
                nn.Conv2d(in_channels=16*4+4, out_channels=32, kernel_size=(5,5), padding=2, padding_mode='circular'),
            ]),
            nn.Conv2d(in_channels=32*3, out_channels=32, kernel_size=(1,1), padding=0, padding_mode='circular'),
            nn.Conv2d(in_channels=32,   out_channels=16, kernel_size=(1,1), padding=0, padding_mode='circular'),
            nn.Conv2d(in_channels=16,   out_channels=1,  kernel_size=(1,1), padding=0, padding_mode='circular'),
        ])


    def forward(self, x):
        x = self.cast_inputs(x)
        x = self.relu1(x)
        input = self.features(x)
        for n, layer in enumerate(self.layers):
            if isinstance(layer, nn.ModuleList):
                x = self.features(x)
                x = torch.cat([ x, input ], dim=1) if n != 0 else x
                channels = [ sublayer(x) for sublayer in layer ]
                x = torch.cat(channels, dim=1)
            else:
                x = layer(x)
            x = self.activation(x)
        x = self.relu1(x)
        return x


    def loss(self, outputs, expected, inputs):
        """
        GameOfLifeReverseOneGAN() computes the backwards timestep
        discriminator GameOfLifeHardcodedReLU1_21() replays the board again forwards
        forward_loss is the MSE difference between the backwards prediction and forward play
        classic_loss biases the network towards the exact solution, but reduces to zero as forward_loss approaches zero
        sum_loss is a heuristic to guide solution towards the correct cell count and avoid all 0 or all 1 solutions
        binary_loss penalizes non-binary output
        """
        forwards     = self.discriminator(outputs)
        forward_loss = self.criterion(forwards, inputs)                      # loss==0 if forward play matches input
        classic_loss = self.criterion(outputs, expected)                     # loss==0 if dataset matches output
        sum_loss1    = ( torch.mean(forwards) - torch.mean(inputs)   ) ** 2  # loss==0 if cell count is the same
        sum_loss2    = ( torch.mean(outputs)  - torch.mean(expected) ) ** 2  # loss==0 if cell count is the same
        binary_loss1 = torch.mean( 0.5**2 - ( forwards - 0.5 )**2 )          # loss==0 if all values are 1 or 0
        binary_loss2 = torch.mean( 0.5**2 - ( outputs  - 0.5 )**2 )          # loss==0 if all values are 1 or 0
        loss = (
            forward_loss + binary_loss1 + binary_loss2 + sum_loss1
            + F.relu( ( classic_loss + sum_loss2 ) * (forward_loss - 0.01) * 10 )
        )
        return loss

    # noinspection PyTypeChecker
    def accuracy(self, outputs, expected, inputs):
        """ Accuracy here is based upon if the output matches the input after forward play """
        # return super().accuracy(outputs, expected, inputs)
        forwards = self.discriminator(outputs)
        return pt.sum( self.cast_bool(forwards) == self.cast_bool(inputs) ).cpu().numpy() / np.prod(outputs.shape)


    def unfreeze(self: T) -> T:
        if not self.loaded: self.load()
        excluded = { 'discriminator', 'features' }
        for name, parameter in self.named_parameters():
            if not set( name.split('.') ) & excluded:
                parameter.requires_grad = True
        return self


if __name__ == '__main__':
    from neural_networks.train import train
    model = GameOfLifeReverseInception()
    train(model, grid_size=5, reverse_input_output=True)
