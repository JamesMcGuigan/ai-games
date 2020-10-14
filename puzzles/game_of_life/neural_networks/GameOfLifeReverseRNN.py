from abc import ABCMeta
from typing import TypeVar

import numpy as np
import torch
import torch as pt
import torch.nn.functional as F
from torch import nn

from neural_networks.GameOfLifeBase import GameOfLifeBase
from neural_networks.hardcoded.GameOfLifeHardcodedReLU1_21 import GameOfLifeHardcodedReLU1_21
from neural_networks.modules.ReLUX import ReLU1

# noinspection PyTypeChecker
T = TypeVar('T', bound='GameOfLifeFeatures')
class GameOfLifeFeatures(nn.Module):
    """ Count the number of neighbouring cells (excluding self)"""
    def __init__(self):
        super().__init__()
        self.forward_play = GameOfLifeHardcodedReLU1_21()
        self.conv2d       = nn.Conv2d(in_channels=1, out_channels=4, bias=False,
                                      kernel_size=(3,3), padding=1, padding_mode='circular')
        self.conv2d.weight.data = torch.tensor([
            # neighbours
            [[[  1.0,  1.0,  1.0 ],
              [  1.0,  0.0,  1.0 ],
              [  1.0,  1.0,  1.0 ]]],
        ])
        self.conv2d.weight.requires_grad_(False)  # Weights are hardcoded


    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, 1, x.shape[2], x.shape[3])
        features = self.conv2d(x)
        # features = torch.cat([ self.activations[n](features[:,n:n+1,:,:]) for n in range(features.shape[1]) ], dim=1)
        x = torch.cat([
            x,                     # +1 channels
            self.forward_play(x),  # +1 channels
            features,              # +2 channels
        ], dim=1)
        x = x.reshape(shape[0], -1, shape[2], shape[3])
        return x




class GameOfLifeReverseRNN(GameOfLifeBase, metaclass=ABCMeta):
    """
    """
    def __init__(self, grid_size=25, state_size=9):
        super().__init__()
        self.grid_size  = grid_size
        self.state_size = state_size

        # self.criterion = FocalLoss()
        self.criterion    = nn.BCELoss()
        # self.criterion  = nn.MSELoss()
        self.activation   = nn.PReLU()
        # self.activation.weight.data = torch.tensor([-0.5])  # 11N Solution
        self.relu1       = ReLU1()

        # discriminator must be at top-level for autograd to work, its weights are added to the savefile
        self.discriminator = GameOfLifeHardcodedReLU1_21()
        for name, parameter in self.discriminator.named_parameters(): parameter.requires_grad = False

        self.features = GameOfLifeFeatures()

        in_channels  = (2+self.state_size)*3+1
        out_channels = self.state_size+1
        self.cnn_layers = nn.ModuleList([
            # Input, Prediction, State
            nn.Conv2d( in_channels,        out_channels=9,            kernel_size=(3,3), padding=1, padding_mode='circular'),
            nn.Conv2d( in_channels=9+1,    out_channels=9,            kernel_size=(1,1)),
            nn.Conv2d( in_channels=9*3+1,  out_channels=9,            kernel_size=(3,3), padding=1, padding_mode='circular'),
            nn.Conv2d( in_channels=9+1,    out_channels=9,            kernel_size=(1,1)),
            nn.Conv2d( in_channels=9+1,    out_channels=9,            kernel_size=(1,1)),
            nn.Conv2d( in_channels=9+1,    out_channels=out_channels, kernel_size=(1,1)),
        ])
        self.dense_layers = nn.ModuleList([
            # nn.Linear(in_features=grid_size*grid_size*(3+1), out_features=grid_size*grid_size*2),
            # nn.Linear(in_features=grid_size*grid_size*(2+1), out_features=grid_size*grid_size*2),
            # nn.Linear(in_features=grid_size*grid_size*(2+1), out_features=grid_size*grid_size*3)
        ])


    def predict(self, x, max_steps=5, **kwargs):
        output = super().predict(x, max_steps=max_steps, **kwargs)
        output = output.reshape((-1, output.shape[1], output.shape[2], output.shape[3]))
        output = output[:,0,:,:]
        output = output.squeeze()
        return output


    def forward(self, x, max_steps=5, early_stopping=False):
        x = self.cast_inputs(x)
        inputs = x[:,0:1,:,:]
        if x.shape[1] == 1:
            x = torch.cat([
                inputs,                                                                              # Input
                torch.zeros((x.shape[0], 1,               x.shape[2], x.shape[3])).to(self.device),  # Prediction
                torch.zeros((x.shape[0], self.state_size, x.shape[2], x.shape[3])).to(self.device),  # State
            ], dim=1)

        # x[:,0:2,:,:] = self.relu1(x[:,0:2,:,:])  # ReLU1 on input and prediction
        x = self.relu1(x)
        for n in range(max_steps):
            for layer in self.cnn_layers:
                if x.shape[1] == layer.in_channels // 3:
                    x = self.features(x)
                if x.shape[1] == layer.in_channels - 1:
                    x = torch.cat([ inputs, x ], dim=1)

                x = layer(x)
                x = self.activation(x)  # V shaped - SVM filter lines
                # x = self.relu1(x)     # Z shaped - cast back to boolean

            # x = self.features(x)
            shape = x.shape
            x = x.flatten(1)
            for layer in self.dense_layers:
                x = torch.cat([ x, inputs.flatten(1) ], dim=1)
                x = layer(x)
                x = self.activation(x)
            x = x.reshape((shape[0], -1, shape[2], shape[3]))

            x = torch.cat([ inputs, x ], dim=1)      # append input to output
            x = self.relu1(x)  # ReLU1 on input and prediction

            if early_stopping and n != max_steps-1:
                reinputs, prediction, state = x[:,0:1,:,:], x[:,1:2,:,:], x[:,2:,:,:]
                forwards   = self.discriminator(prediction)
                if torch.all(torch.eq( inputs, forwards )):
                    break

        return x


    def cell_count_loss(self, boards1, boards2):
        """ Return average difference in cell count (per board) squared """
        return torch.mean((torch.mean(boards1.flatten(1), dim=0) - torch.mean(boards2.flatten(1), dim=0)) ** 2)

    def binary_loss(self, x):
        """ loss == 0 if all values are either 0 or 1; max loss == 0.5 if all values are 0.5 """
        return 0.5**2 - torch.mean( ( x - 0.5 ) ** 2 )
        # return 0.5    - torch.mean( torch.abs( x - 0.5 ) )


    def loss(self, outputs, expected, inputs, max_steps=5):
        """
        GameOfLifeReverseOneGAN() computes the backwards timestep
        discriminator GameOfLifeHardcodedReLU1_21() replays the board again forwards
        forward_loss is the MSE difference between the backwards prediction and forward play
        classic_loss biases the network towards the exact solution, but reduces to zero as forward_loss approaches zero
        sum_loss is a heuristic to guide solution towards the correct cell count and avoid all 0 or all 1 solutions
        binary_loss penalizes non-binary output
        """
        losses = torch.zeros((max_steps,), dtype=torch.float, requires_grad=True).to(self.device)
        for t in range(max_steps):
            if t != 0:
                outputs = torch.cat([ inputs, outputs[:,1:,:,:] ], dim=1)  # reset inputs
                outputs = self(outputs, max_steps=1, early_stopping=False)

            reinputs, prediction, state = outputs[:,0:1,:,:], outputs[:,1:2,:,:], outputs[:,2:,:,:]
            forwards = self.discriminator(prediction)

            forward_loss  = self.criterion(forwards, inputs)   # loss==0 if forward play matches input
            identity_loss = self.criterion(reinputs, inputs)   # loss==0 if forward play matches input

            cell_count_loss_forwards   = self.cell_count_loss(forwards,   inputs)
            cell_count_loss_reinputs   = self.cell_count_loss(reinputs,   inputs)
            cell_count_loss_prediction = self.cell_count_loss(prediction, expected)
            cell_count_loss = (cell_count_loss_forwards + cell_count_loss_reinputs + cell_count_loss_prediction) / 3

            binary_loss_prediction = self.binary_loss(prediction)
            binary_loss_reinputs   = self.binary_loss(reinputs)
            binary_loss = (binary_loss_prediction + binary_loss_reinputs) / 2

            dataset_loss = self.criterion(prediction, expected)       # loss==0 if output matches dataset
            dataset_loss = F.relu( dataset_loss * (torch.tanh(forward_loss) - 0.01) )  # fade out classic loss

            losses[t] = forward_loss + identity_loss + ( cell_count_loss + binary_loss + dataset_loss ) / 3
            losses[t] = losses[t] * (t+1)  # end of sequence outputs are most important

        loss = torch.sum( losses ) / np.sum(np.arange(1,max_steps+1))
        return loss


    # noinspection PyTypeChecker
    def accuracy(self, outputs, expected, inputs):
        """ Accuracy here is based upon if the output matches the input after forward play """
        # return super().accuracy(outputs, expected, inputs)
        reinputs, prediction, state = outputs[:,0:1,:,:], outputs[:,1:2,:,:], outputs[:,2:,:,:]
        forwards = self.discriminator(self.cast_int(prediction))
        return pt.sum( self.cast_bool(forwards) == self.cast_bool(inputs) ).cpu().numpy() / np.prod(outputs.shape)


    def unfreeze(self: T) -> T:
        if not self.loaded: self.load()
        excluded = { 'discriminator', 'features' }
        for name, parameter in self.named_parameters():
            if not set( name.split('.') ) & excluded:
                parameter.requires_grad = True
        return self


if __name__ == '__main__':
    from neural_networks.train import train
    model = GameOfLifeReverseRNN(grid_size=25)
    train(model, batch_size=25, grid_size=5, reverse_input_output=True)
    # model.print_params()
