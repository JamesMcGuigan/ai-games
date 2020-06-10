
# Source: https://codereview.stackexchange.com/questions/132914/crop-black-border-of-image-using-numpy
import numpy as np

from src.util.np_cache import np_cache



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

