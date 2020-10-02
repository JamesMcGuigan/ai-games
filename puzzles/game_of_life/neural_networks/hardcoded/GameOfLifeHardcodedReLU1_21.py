# The purpose of this neural network is to predict the previous T=-1 state from a game of life board

import torch
import torch.nn as nn

from neural_networks.hardcoded.GameOfLifeHardcoded import GameOfLifeHardcoded
from neural_networks.modules.ReLUX import ReLU1


class GameOfLifeHardcodedReLU1_21(GameOfLifeHardcoded):
    """
    This uses ReLU1 as binary true/false activation layer to implement the game of life rules using 2 nodes:
    AND(
        z3.AtLeast( past_cell, *past_neighbours, 3 ): n >= 3
        z3.AtMost(             *past_neighbours, 3 ): n <  4
    )

    See GameOfLifeHardcodedReLU1_41 for an alternative implementation using 4 nodes
    """
    def __init__(self):
        super().__init__()

        self.input   = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), bias=False)  # no-op trainable layer
        self.counter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(3,3),
                                  padding=1, padding_mode='circular', bias=False)
        self.logics  = nn.ModuleList([
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(1,1))
        ])
        self.output  = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(1,1))
        self.activation = ReLU1()


    def load(self):
        super().load()

        self.input.weight.data   = torch.tensor([[[[1.0]]]])
        self.counter.weight.data = torch.tensor([
            [[[ 0.0, 0.0, 0.0 ],
              [ 0.0, 1.0, 0.0 ],
              [ 0.0, 0.0, 0.0 ]]],

            [[[ 1.0, 1.0, 1.0 ],
              [ 1.0, 0.0, 1.0 ],
              [ 1.0, 1.0, 1.0 ]]]
        ])

        self.logics[0].weight.data = torch.tensor([
            [[[ 1.0 ]], [[  1.0 ]]],   # n >= 3   # z3.AtLeast( past_cell, *past_neighbours, 3 ),
            [[[ 0.0 ]], [[ -1.0 ]]],   # n <  4   # z3.AtMost(             *past_neighbours, 3 ),
        ])
        self.logics[0].bias.data = torch.tensor([
            -3.0   + 1.0,             # n >= 3   # z3.AtLeast( past_cell, *past_neighbours, 3 ),
            +(9-4) - 1.0,             # n <  4   # z3.AtMost(             *past_neighbours, 3 ),
        ])

        # Both of the statements need to be true, and ReLU enforces we can't go above 1
        self.output.weight.data = torch.tensor([[
            [[  1.0 ]],
            [[  1.0 ]],
        ]])
        self.output.bias.data = torch.tensor([ -2.0 + 1.0 ])  # sum() >= 2

        self.to(self.device)
        return self



if __name__ == '__main__':
    from neural_networks.train import train
    import numpy as np

    model = GameOfLifeHardcodedReLU1_21()

    board = np.array([
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,1,1,1,0],
        [0,0,0,0,0],
        [0,0,0,0,0],
    ])
    result1 = model.predict(board)
    result2 = model.predict(result1)
    # assert np.array_equal(board, result2)

    train(model)

    print('-' * 20)
    print(model.__class__.__name__)
    for name, parameter in sorted(model.named_parameters(), key=lambda pair: pair[0].split('.')[0] ):
        print(name)
        print(parameter.data.squeeze().cpu().numpy())
        print()


