from typing import TypeVar

import torch
import torch.nn as nn

from neural_networks.GameOfLifeBase import GameOfLifeBase

# noinspection PyTypeChecker
T = TypeVar('T', bound='GameOfLifeForward_2N')
class GameOfLifeForward_2N(GameOfLifeBase):
    """
    This implements the life_step() function as a Neural Network function
    Training/tested to 100% accuracy over 100,000 random boards

    This is similar to GameOfLifeForward_2, but without the passthrough of the original input
    """
    def __init__(self):
        super().__init__()

        # epoch: 240961 | board_count: 6024000 | loss: 0.0000000000 | accuracy = 1.0000000000 | time: 0.611ms/board
        # Finished Training: GameOfLifeForward_128 - 240995 epochs in 3569.1s
        self.layers = nn.ModuleList([
            # Previous pixel state requires information from distance 2, so we need two 3x3 convolutions
            nn.Conv2d(in_channels=1,    out_channels=2,    kernel_size=(3,3), padding=1, padding_mode='circular'),
            nn.Conv2d(in_channels=2,    out_channels=2,    kernel_size=(1,1)),
            nn.Conv2d(in_channels=2,    out_channels=1,    kernel_size=(1,1)),
        ])
        self.dropout    = nn.Dropout(p=0.0)
        self.activation = nn.PReLU()


    def forward(self, x):
        x = input = self.cast_inputs(x)
        for n, layer in enumerate(self.layers):
            ### Disable Passthrough
            # if layer.in_channels > 1 and layer.in_channels % 2 == 1:   # autodetect 1+in_channels == odd number
            #     x = torch.cat([ x, input ], dim=1)                     # passthrough original cell state
            x = layer(x)
            if n != len(self.layers)-1:
                x = self.activation(x)
                x = self.dropout(x)
            else:
                x = torch.sigmoid(x)  # output requires sigmoid activation
        return x


    def load(self: T) -> T:
        self.loaded = True    # prevent any infinite if self.loaded loops
        self.apply(self.weights_init)
        self.to(self.device)  # ensure all weights, either loaded or untrained are moved to GPU
        # self.eval()           # default to production mode - disable dropout
        # self.freeze()         # default to production mode - disable training
        return self


    def weights_init(self, layer):
        ### Default initialization seems to work best, at least for Z shaped ReLU1 - see GameOfLifeHardcodedReLU1_21.py
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
            ### kaiming_normal_ corrects for mean and std of the relu function
            ### xavier_normal_ works better for ReLU6 and Z shaped activations
            if isinstance(self.activation, (nn.ReLU, nn.LeakyReLU, nn.PReLU)):
                nn.init.kaiming_normal_(layer.weight)
                # nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    # small positive bias so that all nodes are initialized
                    nn.init.constant_(layer.bias, 0.1)
        else:
            # Use default initialization
            pass



if __name__ == '__main__':
    from neural_networks.train import train
    import numpy as np

    model = GameOfLifeForward_2N().load()
    model.print_params()
    print('-' * 20)


    board = np.array([
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,1,1,1,0],
        [0,0,0,0,0],
        [0,0,0,0,0],
    ])
    result1 = model.predict(board)
    result2 = model.predict(result1)

    train(model, batch_size=100, grid_size=5, accuracy_count=100_000)

    result3 = model.predict(board)
    result4 = model.predict(result3)
    assert np.array_equal(board, result4)

    print('-' * 20)
    model.print_params()
