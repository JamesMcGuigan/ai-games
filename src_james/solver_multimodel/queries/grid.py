import numpy as np

# from skimage.measure import block_reduce
# from numpy_lru_cache_decorator import np_cache  # https://gist.github.com/Susensio/61f4fee01150caaac1e10fc5f005eb75
from src_james.util.np_cache import np_cache


def query_true(grid,x,y):          return True
def query_not_zero(grid,x,y):      return grid[x,y]
def query_color(grid,x,y,color):   return grid[x,y] == color


# evaluation/15696249.json - max(1d.argmax())
@np_cache
def query_max_color(grid,x,y,exclude_zero=True):
    return grid[x,y] == max_color(grid, exclude_zero)

@np_cache
def max_color(grid, exclude_zero=True):
    bincount = np.bincount(grid.flatten())
    if exclude_zero:
        bincount[0] = np.min(bincount)  # exclude 0
    return bincount.argmax()

@np_cache
def query_min_color(grid,x,y, exclude_zero=True):
    return grid[x,y] == min_color(grid, exclude_zero)

@np_cache
def min_color(grid,exclude_zero=True):
    bincount = np.bincount(grid.flatten())
    if exclude_zero:
        bincount[0] = np.max(bincount)  # exclude 0
    return bincount.argmin()

@np_cache
def query_max_color_1d(grid,x,y,exclude_zero=True):
    return grid[x,y] == max_color_1d(grid)

@np_cache
def max_color_1d(grid,exclude_zero=True):
    return max(
        [ max_color(row,exclude_zero) for row in grid ] +
        [ max_color(col,exclude_zero) for col in np.swapaxes(grid, 0,1) ]
    )

@np_cache
def query_min_color_1d(grid,x,y):
    return grid[x,y] == min_color_1d(grid)

@np_cache
def min_color_1d(grid):
    return min(
        [ min_color(row) for row in grid ] +
        [ min_color(col) for col in np.swapaxes(grid, 0,1) ]
    )

@np_cache
def query_count_colors(grid,x,y):
    return grid[x,y] >= count_colors(grid)

@np_cache
def query_count_colors_row(grid,x,y):
    return x + grid.shape[0]*y <= count_colors(grid)

@np_cache
def query_count_colors_col(grid,x,y):
    return y + grid.shape[1]*x <= count_colors(grid)


@np_cache
def count_colors(grid):
    bincount = np.bincount(grid.flatten())
    return np.count_nonzero(bincount[1:]) # exclude 0

@np_cache
def query_count_squares(grid,x,y):
    return grid[x,y] >= count_squares(grid)

@np_cache
def query_count_squares_row(grid,x,y):
    return x + grid.shape[0]*y <= count_squares(grid)

@np_cache
def query_count_squares_col(grid,x,y):
    return y + grid.shape[1]*x <= count_squares(grid)

@np_cache
def count_squares(grid):
    return np.count_nonzero(grid.flatten())

@np_cache
def grid_unique_colors(grid):
    return np.unique(grid.flatten())
