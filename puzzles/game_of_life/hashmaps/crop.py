# Source: https://www.kaggle.com/jamesmcguigan/game-of-life-repeating-shapes/edit
from typing import Union

import numpy as np


def crop_inner(grid,tol=0):
    mask = grid > tol

    return grid[np.ix_(mask.any(1),mask.any(0))]

def crop_outer(grid,tol=0):
    mask = grid>tol
    m,n  = grid.shape
    mask0,mask1 = mask.any(0),mask.any(1)
    col_start,col_end = mask0.argmax(),n-mask0[::-1].argmax()
    row_start,row_end = mask1.argmax(),m-mask1[::-1].argmax()
    return grid[row_start:row_end,col_start:col_end]


def crop_and_center(board: np.ndarray, shape=(25,25)) -> Union[np.ndarray, None]:
    cropped = crop_outer(board)
    offset  = ( (shape[0]-cropped.shape[0])//2, (shape[1]-cropped.shape[1])//2 )
    zeros   = np.zeros(shape, dtype=np.int8)
    zeros[ offset[0]:offset[0]+cropped.shape[0], offset[1]:offset[1]+cropped.shape[1] ] = cropped
    return zeros


def filter_crop_and_center(board: np.ndarray, max_size=6, shape=(25,25)) -> Union[np.ndarray, None]:
    for _ in range(2):
        cropped = crop_outer(board)
        if cropped.shape != crop_inner(cropped).shape: continue  # exclude multi-piece shapes
        if cropped.shape[0] <= max_size and cropped.shape[1] <= max_size:
            offset = ( (shape[0]-cropped.shape[0])//2, (shape[1]-cropped.shape[1])//2 )
            zeros  = np.zeros(shape)
            zeros[ offset[0]:offset[0]+cropped.shape[0], offset[1]:offset[1]+cropped.shape[1] ] = cropped
            return zeros
        else:
            # roll viewpoint and try again
            board = np.roll(np.roll(board, shape[0]//2, axis=0), shape[1]//2, axis=1)
    return None
