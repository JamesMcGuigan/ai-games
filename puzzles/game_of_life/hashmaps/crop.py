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

def crop_outer_3d(solution_3d: np.ndarray, tol=0) -> np.ndarray:
    assert len(solution_3d.shape) == 3
    size_t,size_x,size_y = solution_3d.shape

    mask_t = np.array([ np.any(grid) for grid in solution_3d ])
    mask_x = np.any([ grid.any(axis=0) for grid in solution_3d ], axis=0)
    mask_y = np.any([ grid.any(axis=1) for grid in solution_3d ], axis=0)

    t_start,   t_end   = mask_t.argmax(), size_t - mask_t[::-1].argmax()
    col_start, col_end = mask_x.argmax(), size_x - mask_x[::-1].argmax()
    row_start, row_end = mask_y.argmax(), size_y - mask_y[::-1].argmax()
    output = solution_3d[ t_start:t_end, col_start:col_end, row_start:row_end ]
    return output



def crop_and_center(board: np.ndarray, shape=(25,25)) -> Union[np.ndarray, None]:
    cropped = crop_outer(board)
    offset  = ( (shape[0]-cropped.shape[0])//2, (shape[1]-cropped.shape[1])//2 )
    zeros   = np.zeros(shape, dtype=np.int8)
    zeros[ offset[0]:offset[0]+cropped.shape[0], offset[1]:offset[1]+cropped.shape[1] ] = cropped
    return zeros

def crop_and_center_3d(solution_3d: np.ndarray, shape=(25,25)) -> Union[np.ndarray, None]:
    cropped = crop_outer_3d(solution_3d)
    offset  = ( (shape[0]-cropped[0].shape[0])//2, (shape[1]-cropped[0].shape[1])//2 )
    zeros   = np.zeros((cropped.shape[0], *shape), dtype=np.int8)
    zeros[ :, offset[0]:offset[0]+cropped[0].shape[0], offset[1]:offset[1]+cropped[0].shape[1] ] = cropped
    return zeros



def filter_crop_and_center(board: np.ndarray, max_size=6, shape=(25,25)) -> Union[np.ndarray, None]:
    for _ in range(2):
        cropped = crop_outer(board)
        if ( cropped.shape    == crop_inner(cropped).shape  # exclude multi-piece shapes
         and cropped.shape[0] <= max_size and cropped.shape[1] <= max_size
        ):
            offset = ( (shape[0]-cropped.shape[0])//2, (shape[1]-cropped.shape[1])//2 )
            zeros  = np.zeros(shape, dtype=np.int)
            zeros[ offset[0]:offset[0]+cropped.shape[0], offset[1]:offset[1]+cropped.shape[1] ] = cropped
            return zeros
        else:
            # roll viewpoint and try again
            board = np.roll(np.roll(board, shape[0]//2, axis=0), shape[1]//2, axis=1)
    return None


def pad_board(board: np.ndarray, padding=1):
    padded_size = (board.shape[0] + padding*2), (board.shape[1] + padding*2)
    zeros       = np.zeros(padded_size, dtype=np.int8)
    zeros[ padding:-padding, padding:-padding ] = board
    return zeros


def roll_2d(board: np.array, shift: int = 25//2) -> np.array:
    return np.roll( np.roll(board, shift, axis=0), shift, axis=1 )
