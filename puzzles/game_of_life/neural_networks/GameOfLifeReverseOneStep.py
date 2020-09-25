import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_networks.device import device
from neural_networks.GameOfLifeBase import GameOfLifeBase


class GameOfLifeReverseOneStep(GameOfLifeBase):
    """
    This implements the life_step() inverse  function as a Neural Network function
    This model of trying to predict exact inputs plateaus at around 84% accuracy (~0.0063 MSE loss)
    """
    def __init__(self):
        super().__init__()

        self.conv33_1 = nn.Conv2d(in_channels=1,   out_channels=32,  kernel_size=(3,3), padding=1, padding_mode='circular')
        self.conv33_2 = nn.Conv2d(in_channels=32,  out_channels=32,  kernel_size=(3,3), padding=1, padding_mode='circular')
        self.conv11_1 = nn.Conv2d(in_channels=32,  out_channels=16,  kernel_size=(1,1))
        self.conv11_2 = nn.Conv2d(in_channels=16,  out_channels=8,   kernel_size=(1,1))
        self.conv11_3 = nn.Conv2d(in_channels=1+8, out_channels=1,   kernel_size=(1,1))
        self.layers   = [ self.conv33_1, self.conv33_2, self.conv11_1, self.conv11_2, self.conv11_3 ]
        self.dropout  = nn.Dropout(p=0.1)

    def forward(self, x):
        x = input = self.cast_inputs(x)
        for n, layer_fn in enumerate(self.layers):
            if n == len(self.layers)-1:
                x = torch.cat([ x, input ], dim=1)  # passthrough center cell state to each layer except first
            x = layer_fn(x)
            if n != len(self.layers)-1:
                x = F.leaky_relu(x)
                x = self.dropout(x)
            else:
                x = torch.sigmoid(x)  # output requires sigmoid activation
        return x



gameOfLifeReverseOneStep = GameOfLifeReverseOneStep()
gameOfLifeReverseOneStep.to(device)
gameOfLifeReverseOneStep.eval()      # disable dropout


if __name__ == '__main__':
    # PYTHONUNBUFFERED=1 time python3 ./neural_networks/GameOfLifeReverseOneStep.py | tee ./neural_networks/models/GameOfLifeReverseOneStep.log
    from neural_networks.train import train

    train(gameOfLifeReverseOneStep, reverse_input_output=True)
