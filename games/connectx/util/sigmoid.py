import numpy as np

def sigmoid(self, value: float):
    # self.sigmoid_width == 2 means a heuristic score of +-2 will return +-0.73
    # 1 / (1 + math.exp(-(+np.inf))) == 1.0
    # 1 / (1 + math.exp(-(4.0)))     == 0.99
    # 1 / (1 + math.exp(-(4.0)))     == 0.98
    # 1 / (1 + math.exp(-(3.0)))     == 0.95
    # 1 / (1 + math.exp(-(2.0)))     == 0.88
    # 1 / (1 + math.exp(-(1.0)))     == 0.73
    # 1 / (1 + math.exp(-(0.5)))     == 0.62
    # 1 / (1 + math.exp(-(0.0)))     == 0.5
    # 1 / (1 + math.exp(-(-np.inf))) == 0.0
    return 1 / (1 + np.exp(-value))


def scaled_sigmoid(value: float, sigmoid_width: float, sigmoid_height: float = 1.0) -> float:
    # sigmoid_width  == 2 means a heuristic score of +-2 will return +-0.73
    # sigmoid_height is used to distinguish between gameover() == +-1 and heuristic values
    return sigmoid_height * (1 / (1 + np.exp(-value / sigmoid_width))) if sigmoid_width else 0
