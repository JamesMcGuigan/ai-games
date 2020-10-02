# The purpose of this neural network is to predict the previous T=-1 state from a game of life board

import torch
import torch.nn as nn

from neural_networks.hardcoded.GameOfLifeHardcoded import GameOfLifeHardcoded


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

        self.input   = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), bias=False)  # no-op trainable layer
        self.counter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(3,3),
                                  padding=1, padding_mode='circular', bias=False)
        self.logics  = nn.ModuleList([
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(1,1))
        ])
        self.output  = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(1,1))
        self.activation = nn.LeakyReLU()


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


if __name__ == '__main__':
    from neural_networks.train import train
    import numpy as np

    model = GameOfLifeHardcodedLeakyReLU()

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


