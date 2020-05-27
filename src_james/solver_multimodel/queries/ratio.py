from itertools import chain
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
from fastcache._lrucache import clru_cache

from src_james.core.DataModel import Task
from src_james.util.np_cache import np_cache


@np_cache()
def grid_shape_ratio(grid1: np.ndarray, grid2: np.ndarray) -> Tuple[float,float]:
    try:
        return ( grid2.shape[0] / grid1.shape[0], grid2.shape[1] / grid1.shape[1] )
    except:
        return (0, 0)  # For tests

@clru_cache(maxsize=None)
def task_grids(task) -> List[np.ndarray]:
    grids = []
    for test_train in ['test','train']:
        for spec in task[test_train]:
            grids += [ spec.get('input',[]), spec.get('output',[]) ]  # tests not gaurenteed to have outputs
    return grids

@clru_cache(maxsize=None)
def task_grid_shapes(task) -> List[Tuple[int,int]]:
    return [ np.array(grid).shape for grid in task_grids(task) ]

@clru_cache(maxsize=None)
def task_grid_max_dim(task: Task) -> int:
    return max(chain(*task_grid_shapes(task)))

@clru_cache(maxsize=None)
def is_task_shape_ratio_unchanged(task: Task) -> bool:
    return task_shape_ratios(task) == [ (1,1) ]

@clru_cache(maxsize=None)
def is_task_shape_ratio_consistant(task: Task) -> bool:
    return len(task_shape_ratios(task)) == 1

@clru_cache(maxsize=None)
def is_task_shape_ratio_integer_multiple(task: Task) -> bool:
    ratios = task_shape_ratios(task)
    return all([ isinstance(d, int) or d.is_integer() for d in chain(*ratios) ])

@clru_cache(maxsize=None)
def task_shape_ratios(task: Task) -> List[Tuple[float,float]]:
    ratios = list(set([
        grid_shape_ratio(problem.get('input',[]), problem.get('output',[]))
        for problem in task['train']
    ]))
    # ratios = set([ int(ratio) if ratio.is_integer() else ratio for ratio in chain(*ratios) ])
    return ratios

@clru_cache(maxsize=None)
def task_shape_ratio(task: Task) -> Union[Tuple[float,float],None]:
    ratios = task_shape_ratios(task)
    if len(ratios) != 1: return None
    return ratios[0]

@clru_cache(maxsize=None)
def is_task_shape_ratio_consistent(task: Task) -> bool:
    return len(task_shape_ratios(task)) == 1

@clru_cache(maxsize=None)
def is_task_shape_ratio_integer_multiple(task: Task) -> bool:
    ratios = task_shape_ratios(task)
    return all([ isinstance(d, int) or d.is_integer() for d in chain(*ratios) ])

@clru_cache(maxsize=None)
def task_output_grid_shapes(task: Task) -> List[Tuple[int,int]]:
    return list(set(
        problem['output'].shape
        for problem in task['train']
        if problem['output'] is not None
    ))

# TODO: Replace with OutputGridSizeSolver().predict()
@clru_cache(maxsize=None)
def task_output_grid_shape(task) -> Union[Tuple[int,int], None]:
    grid_sizes = task_output_grid_shapes(task)
    return len(grid_sizes) == 1 and grid_sizes[0] or None

@clru_cache(maxsize=None)
def is_task_output_grid_shape_constant(task: Task) -> bool:
    return bool(task_output_grid_shape(task))

