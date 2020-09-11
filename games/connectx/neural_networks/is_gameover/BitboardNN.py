import os
from typing import List

import humanize
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.ConnectXBBNN import bitboard_to_numpy2d
from core.ConnectXBBNN import is_bitboard
from neural_networks.is_gameover.device import device


# noinspection PyAbstractClass
class BitboardNN(nn.Module):
    """Base class for bitboard based NNs, handles casting inputs and save/load functionality"""

    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    def cast(self, x):
        if torch.is_tensor(x):  return x
        if isinstance(x, list): return torch.stack([ self.cast(b) for b in x ])
        if is_bitboard(x):
            x = bitboard_to_numpy2d(x)
        x = torch.from_numpy(x).to(torch.int64)  # int64 required for functional.one_hot()
        x = F.one_hot(x, num_classes=self.one_hot_size)
        x = x.to(torch.float32)  # float32 required for self.fc1(x)
        x = x.to(device)
        return x  # x.shape = (42,3)

    def cast_to_label(self, expected: List[bool]):
        labels = torch.tensor(expected, dtype=torch.float).to(device)  # nn.MSELoss() == float | nn.CrossEntropyLoss() == long
        return labels

    def cast_from_outputs(self, outputs: torch.Tensor) -> np.array:
        """ convert (1,1) tensor back to bool """
        actual = np.array([ bool(round(output)) for output in outputs.data.cpu().numpy().flatten() ])
        return actual

    @property
    def filename(self):
        return os.path.join( os.path.dirname(__file__), f'{self.__class__.__name__}.pth')

    # DOCS: https://pytorch.org/tutorials/beginner/saving_loading_models.html
    def save(self):
        torch.save(self.state_dict(), self.filename)
        print(f'{self.__class__.__name__}.save(): {self.filename} = {humanize.naturalsize(os.path.getsize(self.filename))}')

    def load(self):
        if os.path.exists(self.filename):
            # Ignore errors caused by model size mismatch
            try:
                self.load_state_dict(torch.load(self.filename))
                self.eval()
                print(f'{self.__class__.__name__}.load(): {self.filename} = {humanize.naturalsize(os.path.getsize(self.filename))}')
            except: pass