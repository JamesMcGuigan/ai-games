import numpy as np

from src_james.ensemble.period import get_period_length0
from src_james.ensemble.period import get_period_length1
from src_james.util.np_cache import np_cache


@np_cache
def query_period_length0(grid: np.ndarray, i:int, j:int) -> bool:
    period = get_period_length0(grid)
    result = grid[i,j] == period
    return result


@np_cache
def query_period_length1(grid: np.ndarray, i:int, j:int) -> bool:
    period = get_period_length1(grid)
    result = grid[i,j] == period
    return result
