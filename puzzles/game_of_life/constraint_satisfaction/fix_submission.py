import time

import numpy as np
from numba import njit

from utils.datasets import submission_df
from utils.datasets import submission_file
from utils.datasets import test_df
from utils.game import life_step
from utils.util import csv_to_delta
from utils.util import csv_to_numpy
from utils.util import numpy_to_series


@njit
def is_valid_solution(start: np.ndarray, stop: np.ndarray, delta: int) -> bool:
    # we are rewriting data, so lets double check our work
    test_board = start
    for t in range(delta):
        test_board = life_step(test_board)
    is_valid = np.all( test_board == stop )
    return is_valid


def fix_submission(max_offset=5):
    """
    There was a bug in game_of_life_solver() that was solving to T=-(delta+1) rather than delta.
    The result of this was seemingly random scores on the leaderboard, and extra compute time for an additional delta
    The fix is to play each submission entry forwards, and find the correct delta for start

    See: tests/test_submission.py
    """
    time_start = time.perf_counter()
    idxs  = [ idx for idx in submission_df.index if np.count_nonzero(submission_df.loc[idx]) ]
    stats = {
        "time":    0,
        "empty":   len(submission_df.index) - len(idxs),
        "total":   len(idxs),
        "valid":   0,
        "fixed":   0,
        "invalid": 0
    }
    for idx in idxs:
        delta = csv_to_delta(test_df, idx)
        start = csv_to_numpy(submission_df, idx, key='start')
        stop  = csv_to_numpy(test_df,       idx, key='stop')
        assert np.count_nonzero(start), idx
        assert np.count_nonzero(stop),  idx

        if is_valid_solution(start, stop, delta):
            stats['valid'] += 1
            continue  # submission.csv is valid for idx, so skip

        solution = start
        for t in range(1, max_offset+1):
            solution = life_step(solution)
            if is_valid_solution(solution, stop, delta):
                submission_df.loc[idx] = numpy_to_series(solution, key='start')
                stats['fixed'] += 1
                break
        else:
            submission_df.loc[idx] = numpy_to_series(np.zeros(stop.shape), key='start')  # zero out the entry and retry
            print( f'fix_submission() invalid idx: {idx} | delta: {delta} | cells: {np.count_nonzero(solution != stop)}' )
            stats['invalid'] += 1
            pass
    assert stats['total'] == stats['valid'] + stats['fixed'] + stats['invalid']

    if stats['fixed'] + stats['invalid'] > 0:
        submission_df.sort_index().to_csv(submission_file)
        print( f'fix_submission() wrote: {submission_file}' )
        pass

    time_taken = time.perf_counter() - time_start
    stats['time'] = f'{time_taken:.1f}'
    print('fix_submission()', stats)



if __name__ == '__main__':
    fix_submission()
