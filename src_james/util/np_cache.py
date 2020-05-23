# Inspired by: https://stackoverflow.com/questions/52331944/cache-decorator-for-numpy-arrays/52332109
from functools import wraps

import numpy as np
from fastcache._lrucache import clru_cache


### Profiler: 2x speedup
def np_cache(maxsize=None, typed=True):
    """
        Decorator:
        @np_cache
        def fn(): return value

        @np_cache(maxsize=128, typed=True)
        def fn(): return value
    """
    maxsize_default=None

    def np_cache_generator(function):

        @wraps(function)
        def wrapper(*args, **kwargs):
            ### def encode(*args, **kwargs):
            args = list(args)  # BUGFIX: TypeError: 'tuple' object does not support item assignment
            for i, arg in enumerate(args):
                if isinstance(arg, np.ndarray):
                    hash = arg.tobytes()
                    if hash not in wrapper.cache:
                        wrapper.cache[hash] = arg
                    args[i] = hash
            for key, arg in kwargs.items():
                if isinstance(arg, np.ndarray):
                    hash = arg.tobytes()
                    if hash not in wrapper.cache:
                        wrapper.cache[hash] = arg
                    kwargs[key] = hash

            return cached_wrapper(*args, **kwargs)

        @clru_cache(maxsize=maxsize, typed=typed)
        def cached_wrapper(*args, **kwargs):
            ### def decode(*args, **kwargs):
            args = list(args)  # BUGFIX: TypeError: 'tuple' object does not support item assignment
            for i, arg in enumerate(args):
                if isinstance(arg, bytes) and arg in wrapper.cache:
                    args[i] = wrapper.cache[arg]
            for key, arg in kwargs.items():
                if isinstance(arg, bytes) and arg in wrapper.cache:
                    kwargs[key] = wrapper.cache[arg]

            return function(*args, **kwargs)

        # copy lru_cache attributes over too
        wrapper.cache       = {}
        wrapper.cache_info  = cached_wrapper.cache_info
        wrapper.cache_clear = cached_wrapper.cache_clear

        return wrapper


    ### def np_cache(maxsize=1024, typed=True):
    if callable(maxsize):
        (function, maxsize) = (maxsize, maxsize_default)
        return np_cache_generator(function)
    else:
        return np_cache_generator