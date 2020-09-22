# This code is inspired by my work on the Abstraction and Reasoning Corupus
# https://www.kaggle.com/jamesmcguigan/arc-geometry-solvers

from typing import Callable

import numpy as np

from hashmaps.hash_functions import hash_geometric
from hashmaps.hash_functions import hash_translations


def identity(board): return board
def rot90(board):    return np.rot90(board, 1)
def rot180(board):   return np.rot90(board, 2)
def rot270(board):   return np.rot90(board, 3)
def flip(board):     return np.flip(board)
def flip90(board):   return np.flip(np.rot90(board, 1))
def flip180(board):  return np.flip(np.rot90(board, 2))
def flip270(board):  return np.flip(np.rot90(board, 3))
geometric_transforms = [identity, rot90, rot180, rot270, flip, flip90, flip180, flip270]



def solve_geometric(train_board, test_board) -> Callable:
    """
    Find the function required to correctly orientate train_board to match test_board
    This is a simple brute force search over geometric_transforms until matching hash_translations() are found
    """
    assert hash_geometric(train_board) == hash_geometric(test_board)

    geometric_fn = None
    test_hash    = hash_translations(test_board)
    for transform_fn in geometric_transforms:
        train_transform = transform_fn(train_board)
        train_hash      = hash_translations(train_transform)
        if train_hash == test_hash:
            geometric_fn = transform_fn
            break  # we are lazily assuming there will be only one matching function

    assert geometric_fn is not None
    return geometric_fn


def solve_translation(train_board, test_board) -> Callable:
    """
    Find the function required to correctly transform train_board to match test_board
    We compute the sums of cell counts along each axis, then roll them until they match
    """
    train_x_counts = np.count_nonzero(train_board, axis=1)  # == np.array([ np.count_nonzero(train_board[x,:]) for x in range(train_board.shape[0]) ])
    train_y_counts = np.count_nonzero(train_board, axis=0)  # == np.array([ np.count_nonzero(train_board[:,y]) for y in range(train_board.shape[1]) ])
    test_x_counts  = np.count_nonzero(test_board,  axis=1)  # == np.array([ np.count_nonzero(test_board[x,:])  for x in range(test_board.shape[0])  ])
    test_y_counts  = np.count_nonzero(test_board,  axis=0)  # == np.array([ np.count_nonzero(test_board[:,y])  for y in range(test_board.shape[1])  ])
    assert sorted(train_x_counts) == sorted(test_x_counts)
    assert sorted(train_y_counts) == sorted(test_y_counts)

    # This is a little bit inefficient, compared to comparing indexes of max values, but we are not CPU bound
    x_roll_count = None
    for n in range(len(train_x_counts)):
        if np.roll(train_x_counts, n).tobytes() == test_x_counts.tobytes():
            x_roll_count = n
            break

    y_roll_count = None
    for n in range(len(train_y_counts)):
        if np.roll(train_y_counts, n).tobytes() == test_y_counts.tobytes():
            y_roll_count = n
            break

    assert x_roll_count is not None
    assert y_roll_count is not None

    def transform_fn(board):
        return np.roll(np.roll(board, x_roll_count, axis=0), y_roll_count, axis=1)

    assert np.all( transform_fn(train_board) == test_board )
    return transform_fn


def solve_geometric_translation(train_board, test_board) -> Callable:
    geometric_fn    = solve_geometric(train_board, test_board)
    translation_fn  = solve_translation(geometric_fn(train_board), test_board)

    def transform_fn(board):
        return translation_fn( geometric_fn(board) )
    assert np.all( transform_fn(train_board) == test_board )
    return transform_fn
