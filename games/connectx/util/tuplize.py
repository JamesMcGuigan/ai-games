from struct import Struct

import numpy as np



# noinspection PyTypeChecker
def tuplize(value):
    """
    Recursively cast to an immutable tuple that can be hashed

    >>> tuplize([])
    ()
    >>> tuplize({"a": 1})
    (('a', 1),)
    >>> tuplize({"a": 1})
    (('a', 1),)
    >>> tuplize(np.array([1,2,3]))
    (1, 2, 3)
    >>> tuplize(np.array([[1,2,3],[4,5,6]]))
    ((1, 2, 3), (4, 5, 6))
    >>> tuplize('string')
    'string'
    >>> tuplize(42)
    42
    """
    if isinstance(value, (list,tuple,set)): return tuple(tuplize(v) for v in value)
    if isinstance(value, np.ndarray):
        if len(value.shape) == 1: return tuple(value.tolist())
        else:                     return tuple(tuplize(v) for v in value.tolist())
    if isinstance(value, (dict,Struct)):    return tuple(tuplize(v) for v in value.items())
    return value


if __name__ == '__main__':
    # python3 -m doctest -v util/tuplize.py
    import doctest
    doctest.testmod()
