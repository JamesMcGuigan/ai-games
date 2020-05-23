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
    return x + len(grid.shape[0])*y <= count_colors(grid)

@np_cache
def query_count_colors_col(grid,x,y):
    return y + len(grid.shape[1])*x <= count_colors(grid)


@np_cache
def count_colors(grid):
    bincount = np.bincount(grid.flatten())
    return np.count_nonzero(bincount[1:]) # exclude 0

@np_cache
def query_count_squares(grid,x,y):
    return grid[x,y] >= count_squares(grid)

@np_cache
def query_count_squares_row(grid,x,y):
    return x + len(grid.shape[0])*y <= count_squares(grid)

@np_cache
def query_count_squares_col(grid,x,y):
    return y + len(grid.shape[1])*x <= count_squares(grid)

@np_cache
def count_squares(grid):
    return np.count_nonzero(grid.flatten())

# BROKEN?
@np_cache
def rotate_loop(grid, start=0):
    angle = start
    while True:
        yield np.rot90(grid, angle % 4)
        angle += 1 * np.sign(start)

# BROKEN?
@np_cache
def rotate_loop_rows(grid, start=0):
    angle = start
    while True:
        yield np.rot90(grid, angle % grid.shape[0])
        angle += 1 * np.sign(start)

# BROKEN?
@np_cache
def rotate_loop_cols(grid, start=0):
    angle = start
    while True:
        yield np.rot90(grid, angle % grid.shape[1])
        angle += 1 * np.sign(start)

@np_cache
def flip_loop(grid, start=0):
    angle = start
    while True:
        if angle % 2: yield np.flip(grid)
        else:         yield grid
        angle += 1 * np.sign(start)

# BROKEN?
@np_cache
def flip_loop_rows(grid, start=0):
    angle = start
    while True:
        if angle % grid.shape[0]: yield np.flip(grid)
        else:                     yield grid
        angle += 1 * np.sign(start)

# BROKEN?
@np_cache
def flip_loop_cols(grid, start=0):
    angle = start
    while True:
        if angle % grid.shape[1]: yield np.flip(grid)
        else:                     yield grid
        angle += 1 * np.sign(start)

# BROKEN?
@np_cache
def invert(grid, color=None):
    if callable(color): color = color(grid)
    if color is None:   color = max_color(grid)
    if color:
        grid        = grid.copy()
        mask_zero   = grid[ grid == 0 ]
        mask_square = grid[ grid != 0 ]
        grid[mask_zero]   = color
        grid[mask_square] = 0
    return grid



# Source: https://codereview.stackexchange.com/questions/132914/crop-black-border-of-image-using-numpy
@np_cache
def crop_inner(grid,tol=0):
    mask = grid > tol
    return grid[np.ix_(mask.any(1),mask.any(0))]

@np_cache
def crop_outer(grid,tol=0):
    mask = grid>tol
    m,n  = grid.shape
    mask0,mask1 = mask.any(0),mask.any(1)
    col_start,col_end = mask0.argmax(),n-mask0[::-1].argmax()
    row_start,row_end = mask1.argmax(),m-mask1[::-1].argmax()
    return grid[row_start:row_end,col_start:col_end]

def make_tuple(args):
    if isinstance(args, tuple): return args
    if isinstance(args, list):  return tuple(args)
    return (args,)
