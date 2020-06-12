from struct import Struct

import numpy as np



# noinspection PyTypeChecker
def tuplize(value):
    """Recursively cast to an immutable tuple that can be hashed"""
    if isinstance(value, (list,tuple,set)): return tuple(tuplize(v) for v in value)
    if isinstance(value, np.ndarray):
        if len(value.shape) == 1: return tuple(value.tolist())
        else:                     return tuple(tuplize(v) for v in value.tolist())
    if isinstance(value, (dict,Struct)):    return tuple(tuplize(v) for v in value.items())
    return value
