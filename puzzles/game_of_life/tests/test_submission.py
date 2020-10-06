import numpy as np

from utils.datasets import submission_df
from utils.datasets import test_df
from utils.game import life_step
from utils.util import csv_to_delta
from utils.util import csv_to_numpy


def test_submission_df():
    """
    This test validates that all entires in submission.csv, when played forward match the stop state
    see fix_submission() to correct previously generated data
    """
    failed = []
    for idx in submission_df.index:
        if np.count_nonzero(submission_df.loc[idx]) == 0: continue  # skip empty datapoints

        delta = csv_to_delta(test_df, idx)
        start = csv_to_numpy(submission_df, idx, key='start')
        stop  = csv_to_numpy(test_df,       idx, key='stop')
        assert np.count_nonzero(start), idx
        assert np.count_nonzero(stop),  idx

        # Play the board forward
        board = start
        for t in range(delta):
            board = life_step(board)

        if not np.all( board == stop ):
            print(idx)
            failed.append(idx)
    assert failed == []



if __name__ == '__main__':
    test_submission_df()
