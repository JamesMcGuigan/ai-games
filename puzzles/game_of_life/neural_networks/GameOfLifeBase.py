import os
from typing import Union

import humanize
import numpy as np
import torch
import torch.nn as nn

from neural_networks.device import device


# noinspection PyAbstractClass
class GameOfLifeBase(nn.Module):
    """
    Base class for GameOfLife based NNs
    Handles casting of inputs and model auto save/load functionality
    """

    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")



    ### Load / Save Functionality

    @property
    def filename(self):
        return os.path.join( os.path.dirname(__file__), 'models', f'{self.__class__.__name__}.pth')


    # DOCS: https://pytorch.org/tutorials/beginner/saving_loading_models.html
    def save(self):
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        torch.save(self.state_dict(), self.filename)
        print(f'{self.__class__.__name__}.savefile(): {self.filename} = {humanize.naturalsize(os.path.getsize(self.filename))}')


    def load(self):
        if os.path.exists(self.filename):
            # Ignore errors caused by model size mismatch
            try:
                self.load_state_dict(torch.load(self.filename))
                self.eval()
                self.to(device)
                print(f'{self.__class__.__name__}.load(): {self.filename} = {humanize.naturalsize(os.path.getsize(self.filename))}')
            except: pass



    ### Casting Functions

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
    def cast_inputs(self, x):
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


    def cast_to_numpy(self, x: Union[np.ndarray, torch.Tensor], shape=(25,25)) -> np.ndarray:
        if torch.is_tensor(x):
            return self.cast_to_numpy( x.detach().numpy() )
        elif isinstance(x, np.ndarray):
            if len(x.shape) == 3: return x.reshape((-1, shape[0], shape[1]))
            else:                 return x.reshape(shape)
        else:
            raise TypeError(f'{self.__class__.__name__}.cast_to_numpy() invalid type(x) = {type(x)}')
