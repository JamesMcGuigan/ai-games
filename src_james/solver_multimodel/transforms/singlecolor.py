from typing import Tuple

import numpy as np

from src_james.core.DataModel import Task
from src_james.util.np_cache import np_cache


def identity(input: np.ndarray) -> np.ndarray:
    return input

def np_shape(input: np.ndarray) -> Tuple[int,int]:
    return np.array(input).shape

def np_flatten(input: np.ndarray) -> np.ndarray:
    return np.array(input).flatten()

def np_tobytes(input: np.ndarray) -> bytes:
    return np.array(input).tobytes()

def np_hash(input: np.ndarray) -> int:
    return hash(np.array(input).tobytes())

@np_cache()
def np_bincount(grid: np.ndarray, minlength=11):
    return np.bincount(grid.flatten(), minlength=minlength).tolist()  # features requires a fixed length array

@np_cache()
def unique_colors_sorted(grid: np.ndarray, minsize=11):
    bincount = np_bincount(grid)
    colors   = sorted(np.unique(grid), key=lambda color: bincount[color], reverse=True)
    output   = np.zeros(minsize, dtype=np.int8)
    output[:len(colors)] = colors
    return output

@np_cache()
def task_output_unique_sorted_colors(task: Task):
    grid   = np.concatenate([ problem['output'] for problem in task['train'] ])
    output = unique_colors_sorted(grid)
    return output