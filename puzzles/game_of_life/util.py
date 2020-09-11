import math
from typing import Dict

import numpy as np


def csv_to_delta(df, idx, type='start'):
    return int(df.loc[idx]['delta'])

def csv_to_numpy(df, idx, type='start') -> np.ndarray:
    columns = [col for col in df if col.startswith(type)]
    size    = int(math.sqrt(len(columns)))
    X = df.loc[idx][columns].values
    X = X.reshape((size,size)).astype(np.int8)
    return X

def numpy_to_dict(board: np.ndarray, type='start') -> Dict:
    board  = np.array(board).flatten().tolist()
    output = { f"{type}_{n}": board[n] for n in range(len(board)) }
    return output