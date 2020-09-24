import os
from typing import Union

import humanize
import numpy as np
import torch
import torch.nn as nn

# noinspection PyAbstractClass
from neural_networks.device import device


class GameOfLifeBase(nn.Module):
    """Base class for bitboard based NNs, handles casting inputs and savefile/load functionality"""
    size = 25

    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


    @property
    def filename(self):
        return os.path.join( os.path.dirname(__file__), 'models', f'{self.__class__.__name__}.pth')


    def cast_inputs(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if torch.is_tensor(x):
            return x.to(device)
        if isinstance(x, list):
            x = np.array(x)
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(torch.float32)
            x = x.to(device)
            return x  # x.shape = (42,3)
        raise TypeError(f'{self.__class__.__name__}.cast_to_tensor() invalid type(x) = {type(x)}')


    def cast_to_numpy(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(x, np.ndarray):
            if len(x.shape) == 3: return x.reshape((-1, self.size, self.size))
            else:                 return x.reshape((self.size, self.size))
        elif torch.is_tensor(x):
            return self.cast_to_numpy( x.detach().numpy() )
        else:
            raise TypeError(f'{self.__class__.__name__}.cast_to_numpy() invalid type(x) = {type(x)}')


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
