from __future__ import annotations

import os
import re
from abc import ABCMeta
from typing import TypeVar

import humanize
import torch
import torch.nn as nn



# noinspection PyTypeChecker
T = TypeVar('T', bound='GameOfLifeBase')
class NNBase(nn.Module, metaclass=ABCMeta):
    """
    Base class for GameOfLife based NNs
    Handles: save/autoload, freeze/unfreeze, casting between data formats, and training loop functions
    """
    def __init__(self):
        super().__init__()
        self.loaded  = False  # can't call sell.load() in constructor, as weights/layers have not been defined yet
        self.device  = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


    def __call__(self, *args, **kwargs) -> torch.Tensor:
        if not self.loaded: self.load()  # autoload on first function call
        return super().__call__(*args, **kwargs)



    ### Freeze / Unfreeze

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
        if os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
            return f'./{self.__class__.__name__}.pth'
        else:
            filename = os.path.join( os.path.dirname(__file__), 'models', f'{self.__class__.__name__}.pth' )
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            return filename

    # DOCS: https://pytorch.org/tutorials/beginner/saving_loading_models.html
    def save(self: T, verbose=True) -> T:
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        torch.save(self.state_dict(), self.filename)
        if verbose: print(f'{self.__class__.__name__}.savefile(): {self.filename} = {humanize.naturalsize(os.path.getsize(self.filename))}')
        return self


    def load(self: T, load_weights=True, verbose=True) -> T:
        if load_weights and os.path.exists(self.filename):
            try:
                self.load_state_dict(torch.load(self.filename))
                if verbose: print(f'{self.__class__.__name__}.load(): {self.filename} = {humanize.naturalsize(os.path.getsize(self.filename))}')
            except Exception as exception:
                # Ignore errors caused by model size mismatch
                if verbose: print(f'{self.__class__.__name__}.load(): model has changed dimensions, reinitializing weights\n')
                self.apply(self.weights_init)
        else:
            if verbose:
                if load_weights: print(f'{self.__class__.__name__}.load(): model file not found, reinitializing weights\n')
                # else:          print(f'{self.__class__.__name__}.load(): reinitializing weights\n')
            # self.apply(self.weights_init)

        self.loaded = True    # prevent any infinite if self.loaded loops
        self.to(self.device)  # ensure all weights, either loaded or untrained are moved to GPU
        # self.eval()           # default to production mode - disable dropout
        # self.freeze()         # default to production mode - disable training
        return self



    ### Debugging

    def print_params(self):
        print(self.__class__.__name__)
        print(self)
        for name, parameter in sorted(self.named_parameters(), key=lambda pair: pair[0].split('.')[0] ):
            print(name)
            print(re.sub(r'\n( *\n)+', '\n', str(parameter.data.cpu().numpy())))  # remove extranious newlines
            print()
