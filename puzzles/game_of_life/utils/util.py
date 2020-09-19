from typing import Dict

import numpy as np
import pandas as pd
from fastcache._lrucache import clru_cache


@clru_cache(None)
def csv_column_names(key='start'):
    return [ f'{key}_{n}' for n in range(25**2) ]


def csv_to_delta(df, idx):
    return int(df.loc[idx]['delta'])

def csv_to_delta_list(df):
    return df['delta'].values


def csv_to_numpy(df, idx, key='start') -> np.ndarray:
    try:
        columns = csv_column_names(key)
        board   = df.loc[idx][columns].values
    except:
        board = np.zeros((25, 25))
    board = board.reshape((25,25)).astype(np.int8)
    return board


def csv_to_numpy_list(df, key='start') -> np.ndarray:
    try:
        columns = csv_column_names(key)
        output  = df[columns].values.reshape(-1,25,25)
    except:
        output  = np.zeros((0,25,25))
    return output


# noinspection PyTypeChecker,PyUnresolvedReferences
def numpy_to_dict(board: np.ndarray, key='start') -> Dict:
    assert len(board.shape) == 2  # we want 2D solutions_3d[0] not 3D solutions_3d
    assert key in { 'start', 'stop' }

    board  = np.array(board).flatten().tolist()
    output = { f"{key}_{n}": board[n] for n in range(len(board))}
    return output


def numpy_to_series(board: np.ndarray, key='start') -> pd.Series:
    solution_dict = numpy_to_dict(board, key)
    return pd.Series(solution_dict)


# Source: https://stackoverflow.com/questions/8290397/how-to-split-an-iterable-in-constant-size-chunks
def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


