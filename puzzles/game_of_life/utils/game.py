# Functions for implementing Game of Life Forward Play
from typing import List
from typing import Union

import numpy as np
import scipy.sparse
from joblib import delayed
from joblib import Parallel
from numba import njit


# Source: https://www.kaggle.com/ianmoone0617/reversing-conways-game-of-life-tutorial
def life_step_numpy(X: np.ndarray):
    """Game of life step using generator expressions"""
    nbrs_count = sum(np.roll(np.roll(X, i, 0), j, 1)
                     for i in (-1, 0, 1) for j in (-1, 0, 1)
                     if (i != 0 or j != 0))
    return (nbrs_count == 3) | (X & (nbrs_count == 2))


# Source: https://www.kaggle.com/ianmoone0617/reversing-conways-game-of-life-tutorial
def life_step_scipy(X: np.ndarray):
    """Game of life step using scipy tools"""
    from scipy.signal import convolve2d
    nbrs_count = convolve2d(X, np.ones((3, 3)), mode='same', boundary='wrap') - X
    return (nbrs_count == 3) | (X & (nbrs_count == 2))



# NOTE: @njit doesn't like np.roll(axis=) so reimplement explicitly
@njit
def life_neighbours_xy(board: np.ndarray, x, y, max_value=3):
    size_x = board.shape[0]
    size_y = board.shape[1]
    neighbours = 0
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            if i == j == 0: continue    # ignore self
            xi = (x + i) % size_x
            yj = (y + j) % size_y
            neighbours += board[xi, yj]
            if neighbours > max_value:  # shortcircuit return 4 if overpopulated
                return neighbours
    return neighbours


@njit
def life_neighbours(board: np.ndarray, max_value=3):
    size_x = board.shape[0]
    size_y = board.shape[1]
    output = np.zeros(board.shape, dtype=np.int8)
    for x in range(size_x):
        for y in range(size_y):
            output[x,y] = life_neighbours_xy(board, x, y, max_value)
    return output


@njit
def life_step_njit(board: np.ndarray) -> np.ndarray:
    """Game of life step using generator expressions"""
    size_x = board.shape[0]
    size_y = board.shape[1]
    output = np.zeros(board.shape, dtype=np.int8)
    for x in range(size_x):
        for y in range(size_y):
            cell       = board[x,y]
            neighbours = life_neighbours_xy(board, x, y, max_value=3)
            if ( (cell == 0 and      neighbours == 3 )
              or (cell == 1 and 2 <= neighbours <= 3 )
            ):
                output[x, y] = 1
    return output

life_step = life_step_njit  # create global alias
def life_steps(boards: Union[List[np.ndarray],np.ndarray]) -> List[np.ndarray]:
    """ Parallel version of life_step() but for an array of boards """
    return Parallel(-1)( delayed(life_step)(board) for board in boards )


@njit
def life_step_delta(board: np.ndarray, delta):
    for t in range(delta): board = life_step(board)
    return board


def life_steps_delta(boards: np.ndarray, deltas: Union[np.ndarray,int]) -> np.ndarray:
    """ Parallel version of life_step() but for an array of boards """
    if isinstance(deltas, int): deltas = np.ones((len(boards),)) * deltas
    return np.array( Parallel(-1)( delayed(life_step_delta)(board, delta) for (board, delta) in zip(boards, deltas) ) )


def life_step_3d(board: np.ndarray, delta):
    solution_3d = np.array([ board ], dtype=np.int8)
    for t in range(delta):
        board       = life_step(board)
        solution_3d = np.append( solution_3d, [ board ], axis=0)
    return solution_3d


# RULES: https://www.kaggle.com/c/conway-s-reverse-game-of-life/data
def generate_random_board(shape=(25,25)):
    # An initial board was chosen by filling the board with a random density between 1% full (mostly zeros) and 99% full (mostly ones).
    # DOCS: https://cmdlinetips.com/2019/02/how-to-create-random-sparse-matrix-of-specific-density/
    density = np.random.random() * 0.98 + 0.01
    board   = scipy.sparse.random(*shape, density=density, data_rvs=np.ones).toarray().astype(np.int8)

    # The starting board's state was recorded after the 5 "warmup steps". These are the values in the start variables.
    for t in range(5):
        board = life_step(board)
        if np.count_nonzero(board) == 0:
            return generate_random_board(shape)  # exclude empty boards and try again
    return board

def generate_random_boards(count: int, shape=(25,25)) -> np.ndarray:
    generated_boards = Parallel(-1)( delayed(generate_random_board)(shape) for _ in range(count) )
    return np.array(generated_boards, dtype=np.int8).reshape((-1, *shape))
