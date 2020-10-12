import torch
import torch.nn as nn

from neural_networks.hardcoded.GameOfLifeHardcoded import GameOfLifeHardcoded


class GameOfLifeHardcodedLeakyReLU(GameOfLifeHardcoded):
    """
    LeakyReLU can see in both directions, thus can (sometimes) solve both the logic and output layers
    ReLU1 and tanh has problems with this, however the LeakyReLU solution is less easy for a human to understand
    All activation functions have a hard time solving the counter layer


    LeakyReLU trained solution:
    with sigmoid output:
        logic.weight     [[  -0.002864,   1.094667 ]
                          [ -17.410564, -14.649882 ]]
        logic.bias        [  -3.355898,   9.621474 ]
        output.weight     [  -3.801134,  -6.304538 ]
        output.bias          -2.024391,

    with ReLU1 output (starting with sigmoid weights):
        logics.0.weight  [[   0.029851,   1.03208 ]
                          [ -17.687109, -14.95262 ]]
        logics.0.bias:    [  -3.499570,   9.46179 ]
        output.weight:    [  -3.968801,  -6.47038 ]
        output.bias          -1.865675



    """
    def __init__(self):
        super().__init__()

        self.trainable_layers  = [ 'logics', 'output' ]
        self.input   = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), bias=False)  # no-op trainable layer
        self.counter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(3,3),
                                  padding=1, padding_mode='circular', bias=False)
        self.logics  = nn.ModuleList([
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(1,1))
        ])
        self.output  = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(1,1))
        self.activation = nn.LeakyReLU()


    def load(self, **kwargs):
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
            [ [[   0.2 ]], [[   1.0  ]] ],
            [ [[ -17.9 ]], [[ -15.1  ]] ],
        ])
        self.logics[0].bias.data = torch.tensor([
            -3.6,
             9.1
        ])

        # AND == Both sides need to be positive
        self.output.weight.data = torch.tensor([
            [ [[-4.0 ]], [[-6.5 ]] ],
        ])
        self.output.bias.data = torch.tensor([ -1.8 ])  # Either both Alive or both Dead statements must be true

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

    train(model)

    result3 = model.predict(board)
    result4 = model.predict(result1)
    assert np.array_equal(board, result4)

    print('-' * 20)
    print(model.__class__.__name__)
    for name, parameter in sorted(model.named_parameters(), key=lambda pair: pair[0].split('.')[0] ):
        print(name)
        print(parameter.data.squeeze().cpu().numpy())
        print()


