import numpy as np
import torch
import torch as pt
import torch.nn.functional as F
from torch import nn

from neural_networks.GameOfLifeReverseOneStep import GameOfLifeReverseOneStep
from neural_networks.hardcoded.GameOfLifeHardcodedTanh import GameOfLifeHardcodedTanh


class GameOfLifeReverseOneGAN(GameOfLifeReverseOneStep):
    """
    This implements the life_step() inverse function as a Neural Network function
    Whereas GameOfLifeReverseOneStep achieves 84% accuracy when trained to find the original starting conditions
    GameOfLifeReverseOneGAN uses GameOfLifeForward_128 as a discriminator, to predict any valid input board

    BUG: this still does not train above 84%
    """
    def __init__(self):
        super().__init__()

        # discriminator must be at top-level for autograd to work, its weights are added to the savefile
        self.discriminator = GameOfLifeForward().freeze()

        self.conv33_1 = nn.Conv2d(in_channels=1,    out_channels=256,  kernel_size=(3,3), padding=1, padding_mode='circular')
        self.conv33_2 = nn.Conv2d(in_channels=256,  out_channels=256,  kernel_size=(3,3), padding=1, padding_mode='circular')
        self.conv11_1 = nn.Conv2d(in_channels=256,  out_channels=32,   kernel_size=(1,1))
        self.conv11_2 = nn.Conv2d(in_channels=32,   out_channels=16,   kernel_size=(1,1))
        self.conv11_3 = nn.Conv2d(in_channels=1+16, out_channels=1,    kernel_size=(1,1))
        self.layers   = [ self.conv33_1, self.conv33_2, self.conv11_1, self.conv11_2, self.conv11_3 ]
        self.dropout  = nn.Dropout(p=0.2)



    def loss(self, outputs, expected, inputs):
        """
        GameOfLifeReverseOneGAN() computes the backwards timestep
        discriminator GameOfLifeForward_128() replays the board again forwards
        gan_loss is the MSE difference between the backwards prediction and forward play
        classic_loss biases the network towards the exact solution, but reduces to zero as gan_loss approaches zero
        """
        forwards     = self.discriminator(outputs)
        gan_loss     = self.criterion(forwards, inputs)
        classic_loss = self.criterion(outputs, expected)
        return gan_loss # + (classic_loss * gan_loss)/(classic_loss + gan_loss)

    # noinspection PyTypeChecker
    def accuracy(self, outputs, expected, inputs):
        """ Accuracy here is based upon if the output matches the input after forward play """
        # return super().accuracy(outputs, expected, inputs)
        forwards = self.discriminator(outputs)
        return t.sum( self.bool(forwards) == self.bool(inputs) ).cpu().numpy() / np.prod(outputs.shape)


if __name__ == '__main__':
    from neural_networks.train import train
    model = GameOfLifeReverseOneGAN()
    train(model, reverse_input_output=True)
