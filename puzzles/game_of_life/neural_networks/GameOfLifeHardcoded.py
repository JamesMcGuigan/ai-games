# The purpose of this neural network is to predict the previous T=-1 state from a game of life board
from typing import TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_networks.GameOfLifeBase import GameOfLifeBase

# noinspection PyTypeChecker
T = TypeVar('T', bound='GameOfLifeHardcoded')
class GameOfLifeHardcoded(GameOfLifeBase):
    """
    This implements the life_step() function as a minimalist Neural Network function with hardcoded weights
    Subclasses implement the effect of different activation_fns
    """

    # def load_state_dict(self):
    #     pass

    def save(self):
        return self

    def unfreeze(self: T) -> T:
        super().unfreeze()
        self.freeze()
        for name, parameter in self.named_parameters():
            if name.split('.')[0] in [ 'identity' ]:
                parameter.requires_grad = True
        return self


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



class GameOfLifeHardcodedLeakyReLU(GameOfLifeHardcoded):
    """
    Leaky ReLU trained solution
    logic.weight [[-2.8648230e-03  1.0946677e+00]
                  [-1.7410564e+01 -1.4649882e+01]]
    logic.bias    [-3.3558989  9.621474 ]
    output.weight [-3.8011343 -6.304538 ]
    output.bias   -2.0243912
    """
    def __init__(self):
        super().__init__()

        self.identity = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), bias=False)  # We need at least one trainable layer
        self.counter  = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(3,3),
                                  padding=1, padding_mode='circular', bias=False)
        self.logics   = nn.ModuleList([
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(1,1))
        ])
        self.output   = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(1,1))
        self.activation = nn.LeakyReLU()


    def load(self):
        super().load()

        self.identity.weight.data   = torch.tensor([[[[1.0]]]])
        self.counter.weight.data = torch.tensor([
            [[[ 0.0, 0.0, 0.0 ],
              [ 0.0, 1.0, 0.0 ],
              [ 0.0, 0.0, 0.0 ]]],

            [[[ 1.0, 1.0, 1.0 ],
              [ 1.0, 0.0, 1.0 ],
              [ 1.0, 1.0, 1.0 ]]]
        ])

        self.logics[0].weight.data = torch.tensor([
            [ [[-2.8648230e-03]], [[ 1.0946677e+00]] ],
            [ [[-1.7410564e+01]], [[-1.4649882e+01]] ],
        ])
        self.logics[0].bias.data = torch.tensor([
            -3.3558989,
             9.621474
        ])

        # AND == Both sides need to be positive
        self.output.weight.data = torch.tensor([
            [ [[-3.8011343]], [[-6.304538]] ],
        ])
        self.output.bias.data = torch.tensor([ -2.0243912 ])  # Either both Alive or both Dead statements must be true

        self.to(self.device)
        return self



class ReLU1(nn.Module):
    def forward(self, x):
        return F.relu6(x * 6.0) / 6.0

class ReLUX(nn.Module):
    def __init__(self, scale=1.0):
        super(ReLUX, self).__init__()
        self.scale = float(scale)

    def forward(self, x):
        return F.relu6(x * 6.0/self.scale) / (6.0/self.scale)


class GameOfLifeHardcodedReLU6(GameOfLifeHardcoded):

    def __init__(self):
        super().__init__()

        self.identity = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), bias=False)  # We need at least one trainable layer
        self.counter  = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(3,3),
                                 padding=1, padding_mode='circular', bias=False)
        self.logics   = nn.ModuleList([
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(1,1))
        ])
        self.output     = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=(1,1))
        self.activation = ReLUX(1)


    def load(self):
        super().load()

        self.identity.weight.data   = torch.tensor([[[[1.0]]]])
        self.counter.weight.data = torch.tensor([
            [[[ 0.0, 0.0, 0.0 ],
              [ 0.0, 1.0, 0.0 ],
              [ 0.0, 0.0, 0.0 ]]],

            [[[ 1.0, 1.0, 1.0 ],
              [ 1.0, 0.0, 1.0 ],
              [ 1.0, 1.0, 1.0 ]]]
        ])

        self.logics[0].weight.data = torch.tensor([
            [[[   10.0 ]], [[   1.0 ]]],   # z3.AtMost(             *past_neighbours, 3 ),
            [[[   10.0 ]], [[  -1.0 ]]],   # z3.AtMost(             *past_neighbours, 3 ),
            [[[  -10.0 ]], [[   1.0 ]]],   # z3.AtLeast( past_cell, *past_neighbours, 3 ),
            [[[  -10.0 ]], [[  -1.0 ]]],   # z3.AtLeast( past_cell, *past_neighbours, 3 ),
        ])
        self.logics[0].bias.data = torch.tensor([
             -(10 +  (2)   -1)*1.0,  # Alive + n >= 2   # z3.AtLeast( past_cell, *past_neighbours, 3 ),
             -(10 -  (9-4) +1)*1.0,  # Alive + n <  4   # z3.AtLeast( past_cell, *past_neighbours, 3 ),
             -( 0 +  (3)   -1)*1.0,  # Dead  + n >= 3   # z3.AtMost(             *past_neighbours, 3 ),
             -( 0  - (9-4) +1)*1.0,  # Dead  + n <  4   # z3.AtMost(             *past_neighbours, 3 ),
        ])

        # AND == Both sides need to be positive
        self.output.weight.data = torch.tensor([[
            [[  1.0 ]],
            [[  1.0 ]],
            [[  1.0 ]],
            [[  1.0 ]],
        ]])
        self.output.bias.data = torch.tensor([ -2.0 + 1.0 ])  # Either both Alive or both Dead statements must be true

        self.to(self.device)
        return self



if __name__ == '__main__':
    from neural_networks.train import train
    import numpy as np

    models = [
        GameOfLifeHardcodedLeakyReLU(),
        GameOfLifeHardcodedReLU6(),
    ]
    for model in models:
        board = np.array([
            [0,0,0,0,0],
            [0,0,0,0,0],
            [0,1,1,1,0],
            [0,0,0,0,0],
            [0,0,0,0,0],
        ])
        result1 = model.predict(board)
        result2 = model.predict(result1)
        assert np.array_equal(board, result2), (board, result2)

    for model in models:
        train(model)

    for model in models:
        print('-' * 20)
        print(model.__class__.__name__)
        for name, parameter in sorted(model.named_parameters(), key=lambda pair: pair[0].split('.')[0] ):
            print(name)
            print(parameter.data.squeeze().cpu().numpy())
            print()


