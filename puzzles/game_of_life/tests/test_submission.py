import numpy as np

from constraint_satisfaction.z3_solver import game_of_life_solver
from utils.datasets import submission_df
from utils.datasets import test_df
from utils.datasets import train_df
from utils.game import life_step
from utils.util import csv_to_delta
from utils.util import csv_to_numpy


def test_game_of_life_solver_delta():
    """
    This test validates that all entires in submission.csv, when played forward match the stop state
    see fix_submission() to correct previously generated data
    """

    idx   = 0
    delta = csv_to_delta(train_df, idx)
    start = csv_to_numpy(train_df, idx, key='start')
    stop  = csv_to_numpy(train_df, idx, key='stop')
    assert np.count_nonzero(start), idx
    assert np.count_nonzero(stop),  idx

    # Solve backwards then play the board forward again
    z3_solver, t_cells, solution_3d = game_of_life_solver(stop, delta)
    board = solution_3d[0]
    for t in range(delta):
        board = life_step(board)

    assert np.all( board == stop ), idx


def test_submission_df():
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
    test_game_of_life_solver_delta()
    test_submission_df()
