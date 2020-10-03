import torch
import torch.nn as nn

from neural_networks.hardcoded.GameOfLifeHardcoded import GameOfLifeHardcoded
from neural_networks.modules.ReLUX import ReLU1


class GameOfLifeHardcodedReLU1_41(GameOfLifeHardcoded):
    """
    This uses ReLU1 as binary true/false activation layer to implement the game of life rules using 4 nodes:
    SUM(
        Alive + neighbours >= 2
        Alive + neighbours <  4
        Dead  + neighbours >= 3
        Dead  + neighbours <  4
    ) >= 2

    Alive! is implemented as -10 weight, which is greater than maximum value of the 3x3-1=8 counter convolution
    sum() >= 2 works because Dead and Alive are mutually exclusive conditions

    See GameOfLifeHardcodedReLU1_21 for an alternative implementation using only 2 nodes
    """

    def __init__(self):
        super().__init__()

        self.trainable_layers  = [ 'input', 'output' ]
        self.input   = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), bias=False)  # no-op trainable layer
        self.counter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(3,3),
                                  padding=1, padding_mode='circular', bias=False)
        self.logics  = nn.ModuleList([
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(1,1))
        ])
        self.output  = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=(1,1))
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
            [[[  10.0 ]], [[  1.0 ]]],  # Alive + neighbours >= 2
            [[[  10.0 ]], [[ -1.0 ]]],  # Alive + neighbours <  4
            [[[ -10.0 ]], [[  1.0 ]]],  # Dead  + neighbours >= 3
            [[[ -10.0 ]], [[ -1.0 ]]],  # Dead  + neighbours <  4
        ])
        self.logics[0].bias.data = torch.tensor([
            -10.0 - 2.0 + 1.0,  # Alive +  neighbours >= 2
            -10.0 + 3.0 + 1.0,  # Alive + -neighbours <= 3
              0.0 - 3.0 + 1.0,  # Dead  +  neighbours >= 3
              0.0 + 3.0 + 1.0,  # Dead  + -neighbours <= 3
        ])

        # Both of the Alive or Dead statements need to be true
        #   sum() >= 2 works here because the -10 weight above makes the two clauses mutually exclusive
        # Otherwise it would require a second layer to formally implement:
        #   OR( AND(input[0], input[1]), AND(input[3], input[4]) )
        self.output.weight.data = torch.tensor([[
            [[  1.0 ]],
            [[  1.0 ]],
            [[  1.0 ]],
            [[  1.0 ]],
        ]])
        self.output.bias.data = torch.tensor([ -2.0 + 1.0 ])  # sum() >= 2

        self.to(self.device)
        return self



if __name__ == '__main__':
    from neural_networks.train import train
    import numpy as np

    model = GameOfLifeHardcodedReLU1_41()

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


