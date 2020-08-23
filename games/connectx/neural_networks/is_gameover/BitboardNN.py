import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.ConnectXBBNN import bitboard_to_numpy2d
from core.ConnectXBBNN import is_bitboard


# torch.cuda.init()
# torch.cuda.set_device(0)


class BitboardNN(nn.Module):
    """Base class for bitboard based NNs, handles casting inputs and save/load functionality"""

    def cast(self, x):
        if is_bitboard(x):
            x = bitboard_to_numpy2d(x).flatten()
        x = torch.from_numpy(x).to(torch.int64)  # int64 required for functional.one_hot()
        x = F.one_hot(x, num_classes=self.one_hot_size)
        x = x.to(torch.float32)  # float32 required for self.fc1(x)
        # x = x.to(self.device)  # TODO: get CUDA GPU working with torch
        return x  # x.shape = (42,3)

    @property
    def filename(self):
        return os.path.join( os.path.dirname(__file__), f'{self.__class__.__name__}.pth')

    # DOCS: https://pytorch.org/tutorials/beginner/saving_loading_models.html
    def save(self):
        torch.save(self.state_dict(), self.filename)

    def load(self):
        if os.path.exists(self.filename):
            self.load_state_dict(torch.load(self.filename))
            self.eval()