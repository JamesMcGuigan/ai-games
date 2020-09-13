#!/usr/bin/env python3
# NOTE: import numba -> import cProfile -> import profile as _pyprofile -> conflicts with filename: profile.py

import inspect
import timeit
from typing import Callable
from typing import List
from typing import Union

import numpy as np

from datasets import train_df
from game import life_step
from game import life_step_1
from game import life_step_2
from hashmaps import hash_geometric
from hashmaps import hash_translations

dataset = np.array( train_df[ train_df.columns[-624:] ] ).reshape(-1, 25, 25)
dataset = dataset[:1000]
assert isinstance( dataset, np.ndarray)

def profile_commands(commands: List[Union[Callable, str]], number=3):
    # if os.environ.get('NUMBA_DISABLE_JIT', 0): return          # don't run if numba is disabled
    timings = []
    for command in commands:
        time_taken = timeit.timeit(command, number=1,      globals=locals())  # warmup
        time_taken = timeit.timeit(command, number=number, globals=locals())
        time_taken = time_taken / number / len(dataset)
        timings.append(time_taken)
        print( f"{1000 * 1000 * time_taken:6.1f}µs - {inspect.getsource(command).strip()}")

def profile_life_step():
    #  42.7µs - lambda: [ life_step(x)    for x in dataset ],  # 2882.0µs without numba
    # 200.1µs - lambda: [ life_step_1(x)  for x in dataset ],
    #  38.7µs - lambda: [ life_step_2(x)  for x in dataset ],
    return profile_commands([
        lambda: [ life_step(x)    for x in dataset ],
        lambda: [ life_step_1(x)  for x in dataset ],
        lambda: [ life_step_2(x)  for x in dataset ],
    ])

def profile_hash():
    # 444.4µs - lambda: [ hash_geometric(x)     for x in dataset ],  # 31,914.7µs without numba
    # 438.4µs - lambda: [ hash_translations(x)  for x in dataset ],  # 24,112.0µs without numba
    return profile_commands([
        lambda: [ hash_geometric(x)     for x in dataset ],
        lambda: [ hash_translations(x)  for x in dataset ],
    ])

if __name__ == '__main__':
    profile_life_step()
    profile_hash()
