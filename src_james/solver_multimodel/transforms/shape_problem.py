from typing import Tuple

import numpy as np
from fastcache._lrucache import clru_cache

from src_james.core.DataModel import Problem
from src_james.core.DataModel import Task
from src_james.solver_multimodel.transforms.singlecolor import np_resize
from src_james.solver_multimodel.transforms.singlecolor import np_shape


@clru_cache(None)
def task_grid_feature(task: Task) -> np.ndarray:
    output = np.array(task.grids).flatten().astype(np.int8)
    return output



@clru_cache(None)
def problem_max_shape(problem: Problem) -> Tuple[int,int]:
    input_shape  = np_shape(problem['input'])
    output_shape = np_shape(problem['output'])
    max_shape = (
        max(input_shape[0], output_shape[0]),
        max(input_shape[1], output_shape[1]),
    )
    return max_shape

@clru_cache(None)
def problem_min_shape(problem: Problem) -> Tuple[int,int]:
    input_shape  = np_shape(problem['input'])
    output_shape = np_shape(problem['output'])
    max_shape = (
        min(input_shape[0], output_shape[0]),
        min(input_shape[1], output_shape[1]),
    )
    return max_shape


@clru_cache(None)
def problem_pixels(problem: Problem) -> np.ndarray:
    output = [ grid.flatten() for grid in problem.grids ]
    output = np.concatenate(output).flatten().astype(np.int8)
    return output


@clru_cache(None)
def problem_difference_mask(problem: Problem) -> np.ndarray:
    if problem['output'] is None: return np.array([], dtype=np.int8)
    max_shape = problem_max_shape(problem)
    # min_shape = problem_min_shape(problem)
    input     = np_resize(problem['input'],  max_shape)
    output    = np_resize(problem['output'], max_shape)
    mask      = (input != output).astype(np.int8)
    return mask

@clru_cache(None)
def problem_difference_input(problem: Problem) -> np.ndarray:
    if problem['output'] is None: return np.array([], dtype=np.int8)
    max_shape = problem_max_shape(problem)
    mask      = problem_difference_mask(problem)
    input     = np_resize(problem['input'],  max_shape)
    input     = input[mask]
    return input

@clru_cache(None)
def problem_difference_output(problem: Problem) -> np.ndarray:
    if problem['output'] is None: return np.array([], dtype=np.int8)
    max_shape = problem_max_shape(problem)
    mask      = problem_difference_mask(problem)
    output    = np_resize(problem['output'],  max_shape)
    output    = output[mask]
    return output
