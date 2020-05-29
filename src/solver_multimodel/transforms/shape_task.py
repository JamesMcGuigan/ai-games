from typing import List
from typing import Tuple

import numpy as np
from fastcache._lrucache import clru_cache

from src.datamodel.Task import Task
from src.solver_multimodel.transforms.shape_problem import problem_difference_input
from src.solver_multimodel.transforms.shape_problem import problem_difference_mask
from src.solver_multimodel.transforms.shape_problem import problem_difference_output
from src.solver_multimodel.transforms.shape_problem import problem_max_shape


@clru_cache(None)
def task_shapes(task: Task) -> List[Tuple[int,int]]:
    shapes = [ problem_max_shape(problem) for problem in task['train'] ]
    return shapes

@clru_cache(None)
def task_max_shape(task: Task) -> Tuple[int,int]:
    x,y = zip(task_shapes(task))
    return (max(x), max(y))

@clru_cache(None)
def task_min_shape(task: Task) -> Tuple[int,int]:
    x,y = zip(task_shapes(task))
    return (min(x), min(y))

@clru_cache(None)
def task_pixels(task: Task) -> np.ndarray:
    pixels = [ problem['input'].flatten() for problem in task['train'] ]
    pixels = np.concatenate(pixels).flatten()
    return pixels


@clru_cache(None)
def task_difference_mask(task: Task) -> np.ndarray:
    output = np.concatenate([
        problem_difference_mask(problem)
        for problem in task['train']
    ])
    return output

@clru_cache(None)
def task_difference_input(task: Task):
    output = np.concatenate([
        problem_difference_input(problem)
        for problem in task['train']
    ])
    return output

@clru_cache(None)
def task_difference_output(task: Task):
    output = np.concatenate([
        problem_difference_output(problem)
        for problem in task['train']
    ])
    return output
