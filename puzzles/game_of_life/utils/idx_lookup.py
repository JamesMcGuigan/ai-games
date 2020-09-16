import random
from typing import Any
from typing import Callable
from typing import List

import numpy as np
import pandas as pd

from utils.util import csv_to_numpy


def get_unsolved_idxs(df: pd.DataFrame, submision_df: pd.DataFrame, modulo=(1,0), sort_cells=True, sort_delta=False) -> List[int]:
    """ Compare test_df with submision_df and return any idxs without a matching non-zero entry in submision_df """
    idxs = []
    deltas = sorted(df['delta'].unique()) if sort_delta else [0]
    for delta in deltas:  # [1,2,3,4,5]
        # Process in assumed order of difficulty, easiest first
        if sort_delta:
            df = df[ df['delta'] == delta ]                               # smaller deltas are easier
        if sort_cells:
            df = df.iloc[ df.apply(np.count_nonzero, axis=1).argsort() ]  # smaller grids are easier

        # Create list of unsolved idxs
        for idx in df.index:
            if modulo and idx % modulo[0] != modulo[1]: continue  # allow splitting dataset between different servers
            if idx not in submision_df.index or np.count_nonzero(submision_df.loc[idx]) == 0:
                idxs.append(idx)

    if sort_cells == 'random':
        random.shuffle(idxs)
    if sort_cells == 'reverse':
        idxs = list(reversed(idxs))

    assert isinstance(idxs, list)  # BUGFIX: must return list (not generator), else invalid csv entries occur
    return list(idxs)



def get_invarient_idxs(
        start_df: pd.DataFrame,
        stop_df:  pd.DataFrame = None,
        hash_fn:  Callable[[np.ndarray], Any] = None,
        unique = False,
) -> List[int]:
    """
    Compare start_df with stop_df and return idxs for boards that did not change between deltas
    if hash_fn = callable(board) is provided, then this is hash function used to determine identity
    if start_df == train_df, we can reuse the same file with a single argument
    """
    assert callable(hash_fn) or hash_fn is None
    if hash_fn is None: hash_fn = lambda board: np.array(board, dtype=np.int8).tobytes()
    if stop_df is None: stop_df = start_df

    idxs           = sorted( set(start_df.index) & set(stop_df.index) )
    starts         = [ csv_to_numpy(start_df, idx, key='start') for idx in idxs ]
    stops          = [ csv_to_numpy(stop_df,  idx, key='stop')  for idx in idxs ]
    start_hashes   = [ hash_fn(start) for start in starts ]
    stop_hashes    = [ hash_fn(stop)  for stop  in stops  ]

    invarient_idxs = {
        (start_hash if unique else idx): idx
        for idx, start_hash, stop_hash in zip(idxs, start_hashes, stop_hashes)
        if start_hash == stop_hash
    }
    invarient_idxs = list(invarient_idxs.values())
    return invarient_idxs
