# The purpose of this neural network is to predict the previous T=-1 state from a game of life board

import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_networks.device import device
from neural_networks.GameOfLifeBase import GameOfLifeBase


class GameOfLifeForward(GameOfLifeBase):
    """
    This implements the life_step() function as a Neural Network function
    Training/tested to 100% accuracy over 10,000 random boards
    """
    def __init__(self):
        super().__init__()

        # epoch: 240961 | board_count: 6024000 | loss: 0.0000000000 | accuracy = 1.0000000000 | time: 0.611ms/board
        # Finished Training: GameOfLifeForward - 240995 epochs in 3569.1s
        self.conv1   = nn.Conv2d(in_channels=1,     out_channels=128, kernel_size=(3,3), padding=1, padding_mode='circular')
        self.conv2   = nn.Conv2d(in_channels=128,   out_channels=16,  kernel_size=(1,1))
        self.conv3   = nn.Conv2d(in_channels=16,    out_channels=8,   kernel_size=(1,1))
        self.conv4   = nn.Conv2d(in_channels=1+8,   out_channels=1,   kernel_size=(1,1))
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = input = self.cast_inputs(x)

        x = self.conv1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)

        x = torch.cat([ x, input ], dim=1)  # remind CNN of the center cell state before making final prediction
        x = self.conv4(x)
        x = torch.sigmoid(x)

        return x


gameOfLifeForward = GameOfLifeForward()
gameOfLifeForward.to(device)
gameOfLifeForward.eval()      # disable dropout



if __name__ == '__main__':
    import timeit
    import operator
    from utils.game import generate_random_boards, life_step, life_step_1, life_step_2
    boards  = generate_random_boards(1_000)
    for number in [1,10,100]:
        timings = {
            'gameOfLifeForward() - batch': timeit.timeit(lambda: gameOfLifeForward(boards),                        number=number),
            'gameOfLifeForward() - loop':  timeit.timeit(lambda: [ gameOfLifeForward(board) for board in boards ], number=number),
            'life_step() - njit':          timeit.timeit(lambda: [ life_step(board)         for board in boards ], number=number),
            'life_step() - numpy':         timeit.timeit(lambda: [ life_step_1(board)       for board in boards ], number=number),
            'life_step() - scipy':         timeit.timeit(lambda: [ life_step_2(board)       for board in boards ], number=number),
        }
        print(f'number: {number} | batch_size = {len(boards)}')
        for key, value in sorted(timings.items(), key=operator.itemgetter(1)):
            print(f'{key:27s} = {value/number/len(boards) * 1_000_000:5.1f}µs')
        print()

        # Discovery:
        # neural networks in batch mode can be faster than C compiled classical code, but slower when called in a loop
        # also scipy uses numba @njit under the hood
        #
        # number: 1 | batch_size = 1000
        # gameOfLifeForward() - batch =   7.8µs
        # life_step() - scipy         = 145.2µs
        # life_step() - numpy         = 179.7µs
        # life_step() - njit          = 431.5µs
        # gameOfLifeForward() - loop  = 804.3µs
        #
        # number: 100 | batch_size = 1000
        # gameOfLifeForward() - batch =  29.8µs
        # life_step() - scipy         =  35.6µs
        # life_step() - njit          =  42.8µs
        # life_step() - numpy         = 180.8µs
        # gameOfLifeForward() - loop  = 618.3µs
