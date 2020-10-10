import os
import sys
import time
import traceback
from typing import Tuple

import humanize
import numpy as np
import pandas as pd
from pathos.multiprocessing import ProcessPool

from constraint_satisfaction.fix_submission import is_valid_solution_3d
from constraint_satisfaction.z3_solver import game_of_life_solver
from utils.datasets import submission_file
from utils.datasets import test_df
from utils.datasets import timeout_file
from utils.idx_lookup import get_unsolved_idxs
from utils.plot import plot_3d
from utils.plot import plot_idx
from utils.util import csv_to_delta
from utils.util import csv_to_numpy
from utils.util import numpy_to_dict


# Parallel(n_jobs=n_jobs)([ delayed(solve_board_deltaN)(board, delta, idx) ])
def solve_board_idx(board: np.ndarray, delta: int, idx: int, timeout=0, verbose=True) -> Tuple[np.ndarray, int, float]:
    time_start = time.perf_counter()

    z3_solver, t_cells, solution_3d = game_of_life_solver(board, delta=delta, idx=idx, timeout=timeout, verbose=False)

    time_taken = time.perf_counter() - time_start
    if verbose:
        message = "Solved! " if is_valid_solution_3d(solution_3d) else "unsolved"
        print(f'{idx:05d} | delta = {delta} | cells = {np.count_nonzero(board):3d} -> {np.count_nonzero(solution_3d):3d} | {message} {time_taken:6.1f}s')
    return solution_3d, idx, time_taken


def solve_board_delta1_loop(board: np.ndarray, delta: int, idx: int, timeout=0, verbose=True) -> Tuple[np.ndarray, int, float]:
    """
    instead of trying to solve to delta=5 in one go, what if repeatedly try to solve for delta=1
    turns out this can be significantly faster than trying to solve to delta=N in one go
    """
    time_start = time.perf_counter()

    output = [ board ]
    for t in range(delta):
        z3_solver, t_cells, solution_3d = game_of_life_solver(board, delta=1, idx=idx, timeout=timeout, verbose=False)
        board = solution_3d[0]
        output.insert(0, board)
    solution_3d = np.array(output)

    time_taken = time.perf_counter() - time_start
    if verbose:
        message = "Solved! " if is_valid_solution_3d(solution_3d) else "unsolved"
        print(f'{idx:05d} | delta = {delta} | cells = {np.count_nonzero(board):3d} -> {np.count_nonzero(solution_3d):3d} | {message} {time_taken:6.1f}s')
    return solution_3d, idx, time_taken


def solve_dataframe(
        df: pd.DataFrame = test_df,
        savefile=submission_file,
        timeout=0,
        max_count=0,
        sort_cells=True,
        sort_delta=True,
        modulo=(1,0),
        plot=False,
) -> pd.DataFrame:
    if not os.path.exists(savefile):  open(savefile, 'a').close()
    time_start = time.perf_counter()

    # BUGFIX: game_of_life_solver() previously implemented delta+1, so fix any previous savefiles
    # fix_submission()  # DONE: remove once all datasets have been updated

    timeout_df    = pd.read_csv(timeout_file, index_col='id')
    submission_df = pd.read_csv(savefile,     index_col='id')  # manually copy/paste sample_submission.csv to location
    solved = 0
    total  = 0  # only count number solved in current runtime, ignore history

    # Pathos multiprocessing allows iterator semantics, whilst joblib has reduced CPU usage at the end of each batch
    # 75% CPU load to optimize for localhost CPU cache + solve memory leaks on Kaggle
    cpus = int( os.cpu_count() * (3/4 if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') else 3/4) )
    pool = ProcessPool(ncpus=cpus)
    try:
        # # timeouts for kaggle submissions
        # if timeout:
        #     def raise_timeout(signum, frame): raise TimeoutError    # DOC: https://docs.python.org/3.6/library/signal.html
        #     signal.signal(signal.SIGALRM, raise_timeout)            # Register a function to raise a TimeoutError on the signal.
        #     signal.alarm(timeout)                                   # Schedule the signal to be sent after ``time``.

        idxs     = get_unsolved_idxs(df, submission_df, modulo=modulo, sort_cells=sort_cells, sort_delta=sort_delta)
        idxs     = [ idx for idx in idxs if idx not in timeout_df.index ]  # exclude timeouts
        deltas   = ( csv_to_delta(df, idx)             for idx in idxs )   # generator
        boards   = ( csv_to_numpy(df, idx, key='stop') for idx in idxs )   # generator
        timeouts = ( min(60*60, timeout * 0.99 - (time.perf_counter() - time_start) if timeout else 0)  for _ in idxs )  # generator
        # NOTE: Z3 timeouts are inexact, but will hopefully occur just before the signal timeout

        solution_idx_iter = pool.uimap(solve_board_idx, boards, deltas, idxs, timeouts)
        for solution_3d, idx, time_taken in solution_idx_iter:
            total += 1
            if is_valid_solution_3d(solution_3d):
                solved += 1

                # Reread file, update and persist
                submission_df          = pd.read_csv(savefile, index_col='id')
                solution_dict          = numpy_to_dict(solution_3d[0])
                submission_df.loc[idx] = pd.Series(solution_dict)
                submission_df.sort_index().to_csv(savefile)

                if plot: plot_3d(solution_3d)
            else:
                if plot: plot_idx(df, idx)  # plot failures
                # Record unsat and timeout failures to prevent kaggle getting stuck on unsolved items
                if not timeout or time_taken > timeout/3:
                    timeout_df          = pd.read_csv(timeout_file, index_col='id')
                    timeout_df.loc[idx] = pd.Series({ "id": idx, "timeout": int(time_taken) })
                    timeout_df.sort_values(by='timeout').to_csv(timeout_file)

            # timeouts for kaggle submissions
            if max_count and max_count <= total:                            raise EOFError()
            if timeout   and timeout   <= time.perf_counter() - time_start: raise TimeoutError()
    except (KeyboardInterrupt, EOFError, TimeoutError): pass
    except Exception as exception:
         print('Exception: solve_dataframe(): ', exception)
         traceback.print_exception(*sys.exc_info())
    finally:
        time_taken = time.perf_counter() - time_start
        percentage = (100 * solved / total) if total else 0
        # noinspection PyTypeChecker
        print(f'Solved: {solved}/{total} = {percentage:.1f}% in {humanize.naturaldelta(time_taken)}')

        pool.terminate()
        pool.clear()
    return submission_df
