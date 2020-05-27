import numpy as np

from src_james.solver_multimodel.transforms.singlecolor import np_bincount
from src_james.solver_multimodel.transforms.singlecolor import unique_colors_sorted
from src_james.util.np_cache import np_cache


@np_cache()
def query_bincount(grid: np.ndarray, i:int, j:int, pos: 0) -> bool:
    bincount = np_bincount(grid)
    if len(bincount) >= pos: return False
    result = grid[i,j] == bincount[pos]
    return result

@np_cache()
def query_bincount_sorted(grid: np.ndarray, i:int, j:int, pos: 0) -> bool:
    bincount = unique_colors_sorted(grid)
    if len(bincount) >= pos: return False
    result = grid[i,j] == bincount[pos]
    return result
