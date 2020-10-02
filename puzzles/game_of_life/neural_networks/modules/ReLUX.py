import torch.nn as nn
import torch.nn.functional as F


class ReLU1(nn.Module):
    def forward(self, x):
        return F.relu6(x * 6.0) / 6.0

class ReLUX(nn.Module):
    def __init__(self, max_value: float=1.0):
        super(ReLUX, self).__init__()
        self.max_value = float(max_value)
        self.scale     = 6.0 / self.max_value

    def forward(self, x):
        return F.relu6(x * self.scale) / self.scale
