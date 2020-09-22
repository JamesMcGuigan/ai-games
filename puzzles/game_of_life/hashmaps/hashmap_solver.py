# Kaggle Notebook: https://www.kaggle.com/jamesmcguigan/game-of-life-hashmap-solver/

from collections import defaultdict
from typing import Callable
from typing import List
from typing import Union

import numpy as np
import pandas as pd
from joblib import delayed
from joblib import Parallel

from hashmaps.hash_functions import hash_geometric
from hashmaps.translation_solver import solve_geometric_translation
from utils.datasets import sample_submission_df
from utils.datasets import test_df
from utils.datasets import train_df
from utils.game import life_step
from utils.util import csv_to_delta
from utils.util import csv_to_numpy
from utils.util import csv_to_numpy_list
from utils.util import numpy_to_series


# noinspection PyDefaultArgument
def build_hashmap_database_from_pandas(
        dfs: Union[pd.DataFrame, List[pd.DataFrame]],
        hash_fn: Callable = hash_geometric,
        future_count = 10,
        keys = ['start', 'stop']
):
    boards = extract_boards_from_dataframe(dfs, keys)
    lookup = build_hashmap_database_from_boards(boards, hash_fn=hash_fn, future_count=future_count)
    return lookup


# noinspection PyDefaultArgument
def extract_boards_from_dataframe(dfs: List[pd.DataFrame], keys = ['start', 'stop'] ):
    boards = []
    if not isinstance(dfs, list): dfs = [ dfs ]
    for df in dfs:
        for key in keys:
            for board in csv_to_numpy_list(df, key=key):
                if np.count_nonzero(board) == 0: continue  # skip empty boards
                boards.append(board)
    return boards


def build_hashmap_database_from_boards(
        boards: List[np.ndarray],
        hash_fn: Callable = hash_geometric,
        future_count = 10,
        max_delta    = 5,
):
    assert callable(hash_fn)

    hashmap_database = defaultdict(lambda: defaultdict(dict))  # hashmap_database[stop_hash][delta] = { stop: np, start: np, delta: int }
    future_hashes = Parallel(-1)(
        delayed(build_future_hashes)(board, hash_fn, future_count)
        for board in boards
    )
    for futures, hashes in future_hashes:
        for t in range(len(futures)-max_delta):
            for delta in range(1, max_delta+1):
                start_board = futures[t]
                stop_board  = futures[t + delta]
                stop_hash   = hashes[t + delta]
                hashmap_database[stop_hash][delta] = { 'start': start_board, 'stop': stop_board, 'delta': delta }
    return hashmap_database


def build_future_hashes(board, hash_fn, future_count):
    futures = [ board ]
    for _ in range(future_count): futures.append( life_step(futures[-1]) )
    hashes  = [ hash_fn(board) for board in futures ]
    return futures, hashes


def solve_hashmap_dataframe(hashmap_database=None, verbose=True):
    solved = 0
    failed = 0
    total  = len(test_df.index)
    hashmap_database = hashmap_database or build_hashmap_database_from_pandas([ train_df, test_df ], hash_fn=hash_geometric)

    submission_df = sample_submission_df.copy()
    for test_idx in test_df.index:
        delta       = csv_to_delta(test_df, test_idx)
        test_stop   = csv_to_numpy(test_df, test_idx, key='stop')
        stop_hash   = hash_geometric(test_stop)
        train_start = hashmap_database.get(stop_hash, {}).get(delta, {}).get('start', None)
        train_stop  = hashmap_database.get(stop_hash, {}).get(delta, {}).get('stop', None)
        if train_start is None: continue

        try:
            solution = solve_geometric_translation(train_stop, test_stop)(train_start)

            solution_test = solution
            for t in range(delta): solution_test = life_step(solution_test)
            assert np.all( solution_test == test_stop )

            submission_df.loc[test_idx] = numpy_to_series(solution)
            solved += 1
        except:
            failed += 1

    if verbose:
        print(f'solved = {solved} ({100*solved/total:.1f}%) | failed = {failed} ({100*failed/(solved+failed):.1f}%)')

    return submission_df



if __name__ == '__main__':
    hashmap_database = build_hashmap_database_from_pandas(train_df[:1000], hash_geometric)
    submission_df    = solve_hashmap_dataframe(hashmap_database, verbose=True)
