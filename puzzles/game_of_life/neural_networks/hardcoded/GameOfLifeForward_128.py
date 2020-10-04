import torch
import torch.nn as nn

from neural_networks.GameOfLifeBase import GameOfLifeBase


class GameOfLifeForward_128(GameOfLifeBase):
    """
    This implements the life_step() function as a Neural Network function
    Training/tested to 100% accuracy over 10,000 random boards
    """
    def __init__(self):
        super().__init__()

        # epoch: 240961 | board_count: 6024000 | loss: 0.0000000000 | accuracy = 1.0000000000 | time: 0.611ms/board
        # Finished Training: GameOfLifeForward_128 - 240995 epochs in 3569.1s
        self.conv1   = nn.Conv2d(in_channels=1,     out_channels=128, kernel_size=(3,3), padding=1, padding_mode='circular')
        self.conv2   = nn.Conv2d(in_channels=128,   out_channels=16,  kernel_size=(1,1))
        self.conv3   = nn.Conv2d(in_channels=16,    out_channels=8,   kernel_size=(1,1))
        self.conv4   = nn.Conv2d(in_channels=1+8,   out_channels=1,   kernel_size=(1,1))
        self.dropout = nn.Dropout(p=0.0)
        self.activation = nn.PReLU()

    def forward(self, x):
        x = input = self.cast_inputs(x)

        x = self.conv1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = torch.cat([ x, input ], dim=1)  # remind CNN of the center cell state before making final prediction
        x = self.conv4(x)
        x = torch.sigmoid(x)

        return x


if __name__ == '__main__':
    from neural_networks.train import train
    import numpy as np

    model = GameOfLifeForward_128().load(load_weights=False)
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

    train(model, batch_size=100, grid_size=25, accuracy_count=100_000)

    result3 = model.predict(board)
    result4 = model.predict(result3)
    assert np.array_equal(board, result4)

    print('-' * 20)
    model.print_params()
