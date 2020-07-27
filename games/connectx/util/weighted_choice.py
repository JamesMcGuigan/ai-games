from typing import Union

import numpy as np
from numba import njit



@njit
def weighted_choice(options: np.ndarray, weights: np.ndarray) -> Union[int,float]:
    """ Returns weighted choice from options array given unnormalized weights """
    assert len(options) == len(weights) != 0
    total = np.sum(weights)
    rand  = np.random.rand() * total

    for i in range(len(options)):
        weight = weights[i]
        if weight < rand:
            rand -= weight
        else:
            return options[i]
    return np.random.choice(options)  # backup incase all weights are zero
