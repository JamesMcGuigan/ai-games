from __future__ import annotations

import os
from abc import ABCMeta
from typing import List
from typing import TypeVar
from typing import Union

import humanize
import numpy as np
import torch
import torch.nn as nn

from neural_networks.device import device

# noinspection PyTypeChecker
T = TypeVar('T', bound='GameOfLifeBase')
class GameOfLifeBase(nn.Module, metaclass=ABCMeta):
    """
    Base class for GameOfLife based NNs
    Handles casting, save/autoload, and
    """

    def __init__(self):
        super().__init__()
        self.device    = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.criterion = nn.MSELoss()
        self.loaded    = False  # can't call sell.load() in constructor, as weights/layers have not been defined yet



    ### Prediction

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        if not self.loaded: self.load()
        return super().__call__(*args, **kwargs)

    def predict(self, inputs: Union[List[np.ndarray], np.ndarray, torch.Tensor]) -> np.ndarray:
        """ Wrapper function around __call__() that returns a numpy int8 array for external usage """
        outputs = self(inputs)
        outputs = self.int(outputs).squeeze().cpu().numpy()
        return outputs



    ### Training

    def loss(self, outputs, expected, input):
        return self.criterion(outputs, expected)

    def accuracy(self, outputs, expected, inputs) -> float:
        # noinspection PyTypeChecker
        return torch.sum( self.int(outputs) == self.int(expected) ).cpu().numpy() / np.prod(outputs.shape)



    ### Freee / Unfreeze

    def freeze(self: T) -> T:
        if not self.loaded: self.load()
        for name, parameter in self.named_parameters():
            parameter.requires_grad = False
        return self

    def unfreeze(self: T) -> T:
        if not self.loaded: self.load()
        for name, parameter in self.named_parameters():
            parameter.requires_grad = True
        return self



    ### Load / Save Functionality

    @property
    def filename(self) -> str:
        return os.path.join( os.path.dirname(__file__), 'models', f'{self.__class__.__name__}.pth')


    # DOCS: https://pytorch.org/tutorials/beginner/saving_loading_models.html
    def save(self: T) -> T:
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        torch.save(self.state_dict(), self.filename)
        print(f'{self.__class__.__name__}.savefile(): {self.filename} = {humanize.naturalsize(os.path.getsize(self.filename))}')
        return self


    def load(self: T) -> T:
        if os.path.exists(self.filename):
            try:
                self.load_state_dict(torch.load(self.filename))
                print(f'{self.__class__.__name__}.load(): {self.filename} = {humanize.naturalsize(os.path.getsize(self.filename))}')
            except Exception as exception:
                # Ignore errors caused by model size mismatch
                print(f'{self.__class__.__name__}.load(): model has changed dimensions, discarding saved weights\n')
                pass

        self.loaded = True    # prevent any infinite if self.loaded loops
        self.to(self.device)  # ensure all weights, either loaded or untrained are moved to GPU
        self.eval()           # default to production mode - disable dropout
        self.freeze()         # default to production mode - disable training
        return self



    ### Casting

    def bool(self, x: torch.Tensor) -> torch.Tensor:
        # noinspection PyTypeChecker
        return (x > 0.5)


    def int(self, x: torch.Tensor) -> torch.Tensor:
        return self.bool(x).to(torch.int8)


    def cast_to_tensor(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if torch.is_tensor(x):
            return x.to(device)
        if isinstance(x, list):
            x = np.array(x)
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(torch.float32)
            x = x.to(device)
            return x  # x.shape = (42,3)
        raise TypeError(f'{self.__class__.__name__}.cast_to_tensor() invalid type(x) = {type(x)}')


    # DOCS: https://towardsdatascience.com/understanding-input-and-output-shapes-in-convolution-network-keras-f143923d56ca
    # pytorch requires:    contiguous_format = (batch_size, channels, height, width)
    # tensorflow requires: channels_last     = (batch_size, height, width, channels)
    def cast_inputs(self, x: Union[List[np.ndarray], np.ndarray, torch.Tensor]) -> torch.Tensor:
        x = self.cast_to_tensor(x)
        if len(x.shape) == 1:             # single row from dataframe
            x = x.view(1, 1, torch.sqrt(x.shape[0]), torch.sqrt(x.shape[0]))
        elif len(x.shape) == 2:
            if x.shape[0] == x.shape[1]:  # single 2d board
                x = x.view(1, 1, x.shape[0], x.shape[1])
            else: # rows of flattened boards
                x = x.view(-1, 1, torch.sqrt(x.shape[1]), torch.sqrt(x.shape[1]))
        elif len(x.shape) == 3:                                 # numpy  == (batch_size, height, width)
            x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])   # x.shape = (batch_size, channels, height, width)
        elif len(x.shape) == 4:
            pass  # already in (batch_size, channels, height, width) format, so do nothing
        return x
