from collections import UserList
from typing import Any
from typing import Callable
from typing import List
from typing import Union

import numpy as np

from src.util.np_cache import np_cache


def bind(function_or_value: Union[Callable,Any], *args, **kwargs) -> Callable:
    if callable(function_or_value):
        def _bind(*runtime_args, **runtime_kwargs):
            return function_or_value(*args, *runtime_args, **runtime_kwargs, **kwargs)
        return _bind
    else:
        # noinspection PyUnusedLocal
        def _passthrough(*args, **kwargs):
            return function_or_value
        return _passthrough

@np_cache()
def invoke(function_or_value, *args, **kwargs) -> List[Any]:
    if callable(function_or_value):
        return function_or_value(*args, **kwargs)
    else:
        return function_or_value


def append_flat(iterable: List, *args) -> List:
    if not isinstance(iterable, list):
        iterable = list(iterable)
    for arg in args:
        if isinstance(arg, (list,tuple,set,np.ndarray,UserList)):
            iterable += list(arg)
        else:
            iterable.append(arg)
    return iterable


def flatten_deep(iterable, types=(list,tuple,set,np.ndarray,UserList)) -> List:
    output = []
    for item in iterable:
        if isinstance(item, types):
            output += flatten_deep(item)
        else:
            output.append(item)
    return output