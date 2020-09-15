import os
import time
from typing import Tuple

import numpy as np
import pandas as pd
from pathos.multiprocessing import ProcessPool

from constraint_satisfaction.z3_solver import game_of_life_solver
from utils.datasets import submission_file
from utils.datasets import test_df
from utils.datasets import train_df
from utils.idx_lookup import get_unsolved_idxs
from utils.plot import plot_3d
from utils.util import csv_to_delta
from utils.util import csv_to_numpy
from utils.util import numpy_to_dict


# Parallel(n_jobs=n_jobs)([ delayed(solve_board_idx)(board, delta, idx) ])
def solve_board_idx(board: np.ndarray, delta: int, idx: int, verbose=True) -> Tuple[np.ndarray, int]:
    time_start = time.perf_counter()

    z3_solver, t_cells, solution_3d = game_of_life_solver(board, delta, verbose=False)

    time_taken = time.perf_counter() - time_start
    if verbose:
        message = "Solved! " if np.count_nonzero(solution_3d) else "unsolved"
        print(f'{idx:05d} | delta = {delta} | cells = {np.count_nonzero(board):3d} -> {np.count_nonzero(solution_3d):3d} | {message} {time_taken:6.1f}s')
    return solution_3d, idx


def solve_dataframe(
        df: pd.DataFrame = test_df,
        savefile=submission_file,
        timeout=0,
        max_count=0,
        sort_cells=True,
        sort_delta=True,
        modulo=(1,0)
) -> pd.DataFrame:
    if not os.path.exists(savefile):  open(savefile, 'a').close()
    time_start = time.perf_counter()

    submission_df = pd.read_csv(savefile, index_col='id')  # manually copy/paste sample_submission.csv to location
    solved = 0
    total  = 0  # only count number solved in current runtime, ignore history

    # Pathos multiprocessing allows iterator semantics, whilst joblib has reduced CPU usage at the end of each batche
    cpus = os.cpu_count() * 3//4  # 75% CPU load to optimize for CPU cache
    pool = ProcessPool(ncpus=cpus)
    try:
        idxs   = get_unsolved_idxs(df, submission_df, modulo=modulo, sort_cells=sort_cells, sort_delta=sort_delta)
        deltas = ( csv_to_delta(df, idx)             for idx in idxs )  # generator
        boards = ( csv_to_numpy(df, idx, key='stop') for idx in idxs )  # generator

        solution_idx_iter = pool.uimap(solve_board_idx, boards, deltas, idxs)
        for solution_3d, idx in solution_idx_iter:
            total += 1
            if np.count_nonzero(solution_3d) != 0:
                solved += 1

                # Reread file, update and persist
                submission_df          = pd.read_csv(savefile, index_col='id')
                solution_dict          = numpy_to_dict(solution_3d[0])
                submission_df.loc[idx] = pd.Series(solution_dict)
                submission_df.sort_index().to_csv(savefile)

            # timeouts for kaggle submissions
            if max_count and max_count <= total:                            raise TimeoutError()
            if timeout   and timeout   <= time.perf_counter() - time_start: raise TimeoutError()
    except: pass
    finally:
        time_taken = time.perf_counter() - time_start
        percentage = (100 * solved / total) if total else 0
        print(f'Solved: {solved}/{total} = {percentage}% in {time_taken:.1f}s')

        pool.terminate()
        pool.clear()
    return submission_df



if __name__ == '__main__':
    # With boolean logic gives 2-4x speedup over integer logic
    for df, idx in [
        (train_df, 0),  # delta = 3 - solved  4.2s (boolean logic) | 16.1s (integer logic)
        (train_df, 4),  # delta = 1 - solved  7.6s (boolean logic) | 57.6s (integer logic)
        (train_df, 0),  # delta = 3 - solved  4.0s (boolean logic) | 17.3s (integer logic)
    ]:
        delta    = csv_to_delta(df, idx)
        board    = csv_to_numpy(df, idx, key='stop')
        expected = csv_to_numpy(df, idx, key='start')
        z3_solver, t_cells, solution_3d = game_of_life_solver(board, delta)
        plot_3d(solution_3d)

    # solve_dataframe(train_df, savefile=None)
    # solve_dataframe(test_df, savefile='submission.csv')
