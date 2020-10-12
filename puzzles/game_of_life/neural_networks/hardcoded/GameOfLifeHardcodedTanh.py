from typing import TypeVar

import torch
import torch.nn as nn
from torch import tensor

from neural_networks.hardcoded.GameOfLifeHardcoded import GameOfLifeHardcoded
from neural_networks.modules.ReLUX import ReLU1

# noinspection PyTypeChecker
T = TypeVar('T', bound='GameOfLifeHardcodedTanh')
class GameOfLifeHardcodedTanh(GameOfLifeHardcoded):
    """
    This uses Tanh as binary true/false activation layer to implement the game of life rules using 4 nodes:
    AND(
        z3.AtLeast( past_cell, *past_neighbours, 3 ): n >= 3
        z3.AtMost(             *past_neighbours, 3 ): n <  4
    )

    Tanh() is both applied to input data as well as being the activation function for the output
    ReLU1  is still being used as the activation function for the logic_games layers

    In theory, the idea was that this would hopefully make this implementation more robust when dealing with
    non-saturated inputs (ie 0.8 rather than 1.0).
    A continual smooth gradient may (hopefully) assist OuroborosLife in using this both as a loss function
    and as a hidden layer inside its own CNN layers. I suspect a ReLU gradient of 0 may be causing problems.

    In practice, the trained tanh solution converged to using two different order of magnitude scales,
    similar to the manual implementation GameOfLifeHardcodedReLU1_41.py.

    I am unsure if this is make the algorithm more or less stable to non-saturated 0.8 inputs.
    However the final tanh() will produce mostly saturated outputs.

    Trained solution
        input.weight      1.0
        logics.0.weight  [[  2.163727,  2.2645657  ]
                          [ -0.018100, -0.29718676 ]]
        logics.0.bias     [ -2.189014,  2.1635942  ]
        output.weight     [  8.673016,  9.106407   ]
        output.bias        -15.924878,


    See GameOfLifeHardcodedReLU1_21 for an alternative implementation using only 2 nodes
    """
    def __init__(self):
        super().__init__()

        self.trainable_layers  = [ 'logics', 'outputs' ]
        self.counter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(3,3),
                                  padding=1, padding_mode='circular', bias=False)
        self.logics  = nn.ModuleList([
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(1,1))
        ])
        self.output  = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=(1,1))


    def forward(self, x):
        x = input = self.cast_inputs(x)

        x = torch.tanh(x)    # tanh
        x = self.input(x)    # noop - a single node linear layer - torch needs at least one trainable layer
        x = self.counter(x)  # counter counts above 6, so no ReLU6

        for logic in self.logics:
            x = logic(x)
            x = torch.tanh(x)
            # x = ReLU1()(x)  # ReLU1 is needed to make the logic gates work

        x = self.output(x)
        x = torch.tanh(x)
        x = ReLU1()(x)

        return x


    def load(self: T, **kwargs) -> T:
        super().load()
        # self.input.weight.data   = tensor([[[[ 1.0498226 ]]]])
        self.input.weight.data   = tensor([[[[ 1.0 ]]]])
        self.counter.weight.data = tensor([
            [[[ 0.0, 0.0, 0.0 ],
              [ 0.0, 1.0, 0.0 ],
              [ 0.0, 0.0, 0.0 ]]],

            [[[ 1.0, 1.0, 1.0 ],
              [ 1.0, 0.0, 1.0 ],
              [ 1.0, 1.0, 1.0 ]]]
        ]) / self.activation(tensor([ 1.0 ]))

        self.logics[0].weight.data = tensor([
            [[[  2.077 ]], [[  2.197 ]]],   # n >= 3   # z3.AtLeast( past_cell, *past_neighbours, 3 ),
            [[[ -0.020 ]], [[ -0.250  ]]],  # n <  4   # z3.AtMost(             *past_neighbours, 3 ),
        ])
        self.logics[0].bias.data = tensor([
            -2.022,              # n >= 3   # z3.AtLeast( past_cell, *past_neighbours, 3 ),
             1.978,              # n <= 3   # z3.AtMost(             *past_neighbours, 3 ),
        ])

        # Both of the statements need to be true. Tanh after logics has the domain (-1,1)
        # Weights here also need to be sufficiently large to saturate sigmoid()
        self.output.weight.data = tensor([[
            [[ 9.0 ]],
            [[ 9.0 ]],
        ]])
        self.output.bias.data = tensor([ -16.0 ])  # sum() > 1.5 as tanh()'ed inputs may not be at full saturation

        self.to(self.device)
        return self


if __name__ == '__main__':
    from neural_networks.train import train
    import numpy as np

    model = GameOfLifeHardcodedTanh()

    board = np.array([
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,1,1,1,0],
        [0,0,0,0,0],
        [0,0,0,0,0],
    ])
    result1 = model.predict(board)
    result2 = model.predict(result1)

    train(model, batch_size=1000)

    result3 = model.predict(board)
    result4 = model.predict(result3)
    assert np.array_equal(board, result4)

    print('-' * 20)
    print(model.__class__.__name__)
    for name, parameter in sorted(model.named_parameters(), key=lambda pair: pair[0].split('.')[0] ):
        print(name)
        print(parameter.data.squeeze().cpu().numpy())
        print()
