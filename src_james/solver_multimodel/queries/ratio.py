from itertools import chain
from typing import List

import numpy as np
from fastcache._lrucache import clru_cache

from src_james.core.DataModel import Task
from src_james.util.np_cache import np_cache


@np_cache
def grid_shape_ratio(grid1, grid2):
    try:
        return ( grid2.shape[0] / grid1.shape[0], grid2.shape[1] / grid1.shape[1] )
    except:
        return (0, 0)  # For tests

@clru_cache()
def task_grids(task):
    grids = []
    for test_train in ['test','train']:
        for spec in task[test_train]:
            grids += [ spec.get('input',[]), spec.get('output',[]) ]  # tests not gaurenteed to have outputs
    return grids

@clru_cache()
def task_grid_shapes(task):
    return [ np.array(grid).shape for grid in task_grids(task) ]

@clru_cache()
def task_grid_max_dim(task):
    return max(chain(*task_grid_shapes(task)))

@clru_cache()
def is_task_shape_ratio_unchanged(task):
    return task_shape_ratios(task) == [ (1,1) ]

@clru_cache()
def is_task_shape_ratio_consistant(task):
    return len(task_shape_ratios(task)) == 1

@clru_cache()
def is_task_shape_ratio_integer_multiple(task):
    ratios = task_shape_ratios(task)
    return all([ isinstance(d, int) or d.is_integer() for d in chain(*ratios) ])

@clru_cache()
def task_shape_ratios(task: Task) -> List:
    ratios = list(set([
        grid_shape_ratio(problem.get('input',[]), problem.get('output',[]))
        for problem in task['train']
    ]))
    # ratios = set([ int(ratio) if ratio.is_integer() else ratio for ratio in chain(*ratios) ])
    return ratios


@clru_cache()
def is_task_shape_ratio_consistent(task):
    return len(task_shape_ratios(task)) == 1

@clru_cache()
def is_task_shape_ratio_integer_multiple(task):
    ratios = task_shape_ratios(task)
    return all([ isinstance(d, int) or d.is_integer() for d in chain(*ratios) ])

