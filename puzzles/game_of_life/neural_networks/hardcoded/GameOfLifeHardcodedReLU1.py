# The purpose of this neural network is to predict the previous T=-1 state from a game of life board

import torch
import torch.nn as nn

from neural_networks.hardcoded.GameOfLifeHardcoded import GameOfLifeHardcoded
from neural_networks.modules.ReLUX import ReLU1


class GameOfLifeHardcodedReLU1(GameOfLifeHardcoded):

    def __init__(self):
        super().__init__()

        self.identity = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), bias=False)  # We need at least one trainable layer
        self.counter  = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(3,3),
                                  padding=1, padding_mode='circular', bias=False)
        self.logics   = nn.ModuleList([
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(1,1))
        ])
        self.output     = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=(1,1))
        self.activation = ReLU1()


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

    model = GameOfLifeHardcodedReLU1()

    board = np.array([
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,1,1,1,0],
        [0,0,0,0,0],
        [0,0,0,0,0],
    ])
    result1 = model.predict(board)
    result2 = model.predict(result1)
    assert np.array_equal(board, result2)

    train(model)

    print('-' * 20)
    print(model.__class__.__name__)
    for name, parameter in sorted(model.named_parameters(), key=lambda pair: pair[0].split('.')[0] ):
        print(name)
        print(parameter.data.squeeze().cpu().numpy())
        print()


