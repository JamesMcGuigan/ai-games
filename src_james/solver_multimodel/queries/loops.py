import numpy as np

from src_james.util.np_cache import np_cache


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
