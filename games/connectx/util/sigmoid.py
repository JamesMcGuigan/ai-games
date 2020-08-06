import numpy as np

def sigmoid(self, value: float):
    # self.heuristic_scale == 2 means a heuristic score of +-2 will return +-0.73
    # 1 / (1 + math.exp(-(+np.inf))) == 1.0
    # 1 / (1 + math.exp(-(2.0)))     == 0.88
    # 1 / (1 + math.exp(-(1.0)))     == 0.73
    # 1 / (1 + math.exp(-(0.5)))     == 0.62
    # 1 / (1 + math.exp(-(0.0)))     == 0.5
    # 1 / (1 + math.exp(-(-np.inf))) == 0.0
    return 1 / (1 + np.exp(-value))


def scaled_sigmoid(value: float, scale: float):
    # scale == 2 means a heuristic score of +-2 will return +-0.73
    return 1 / (1 + np.exp(-value/scale))
