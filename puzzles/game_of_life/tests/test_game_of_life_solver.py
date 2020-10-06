import time

import numpy as np
import pytest

from constraint_satisfaction.fix_submission import is_valid_solution
from constraint_satisfaction.fix_submission import is_valid_solution_3d
from constraint_satisfaction.z3_solver import game_of_life_solver
from utils.datasets import test_df
from utils.datasets import train_df
from utils.game import life_step
from utils.plot import plot_3d
from utils.plot import plot_idx
from utils.util import csv_to_delta
from utils.util import csv_to_numpy


# With boolean logic gives 2-4x speedup over integer logic
#   (train_df, 0), delta = 3+1 - solved  4.2s (boolean logic) | 16.1s (integer logic)
#   (train_df, 4), delta = 1+1 - solved  7.6s (boolean logic) | 57.6s (integer logic)
#   (train_df, 0), delta = 3+1 - solved  4.0s (boolean logic) | 17.3s (integer logic)
# After fixing delta+=1 bug:
# zero_point_distance=1 -> baseline
#     idx =     0 | delta = 3 | cells =   4 ->   9 | valid = True | time =  2.9s
#     idx =     4 | delta = 1 | cells =  65 ->  72 | valid = True | time =  1.0s
#     idx = 50002 | delta = 1 | cells = 152 -> 149 | valid = True | time = 11.8s
#     idx = 50024 | delta = 2 | cells =  28 ->  43 | valid = True | time =  3.0s
#     idx = 50022 | delta = 3 | cells =   6 ->   9 | valid = True | time =  3.2s
#     idx = 98979 | delta = 4 | cells =   4 ->  15 | valid = True | time =  5.0s
#     idx = 58057 | delta = 5 | cells =   4 ->  15 | valid = True | time = 11.5s
#     idx = 90081 | delta = 1 | cells = 130 ->   0 | valid = False | time =  6.2s
# zero_point_distance=2 -> 2*delta slowdown
#     idx =     0 | delta = 3 | cells =   4 ->  11 | valid = True | time =  5.4s
#     idx =     4 | delta = 1 | cells =  65 -> 144 | valid = True | time =  3.1s
#     idx = 50002 | delta = 1 | cells = 152 -> 255 | valid = True | time = 21.7s
#     idx = 50024 | delta = 2 | cells =  28 ->  89 | valid = True | time = 13.8s
#     idx = 50022 | delta = 3 | cells =   6 ->  27 | valid = True | time = 10.5s
#     idx = 98979 | delta = 4 | cells =   4 ->  42 | valid = True | time = 34.8s
#     idx = 58057 | delta = 5 | cells =   4 ->  85 | valid = True | time = 228.9s
#     idx = 90081 | delta = 1 | cells = 130 -> 213 | valid = True | time = 16.4s


@pytest.mark.parametrize("game_of_life_solver", [
    game_of_life_solver,
    # game_of_life_solver_patterns,
])
@pytest.mark.parametrize("idx", [
    90081,  # requires zero_point_distance=2
    50002,
    50024,
    50022,
])
def test_game_of_life_solver_test_df(idx, game_of_life_solver):
    time_start  = time.perf_counter()

    df          = test_df
    delta       = csv_to_delta(df, idx)
    board       = csv_to_numpy(df, idx, key='stop')

    z3_solver, t_cells, solution_3d = game_of_life_solver(board, delta, verbose=False)
    is_valid    = is_valid_solution_3d(solution_3d)

    time_taken  = time.perf_counter() - time_start
    print(f'idx = {idx:5d} | delta = {delta} | cells = {np.count_nonzero(board):3d} -> {np.count_nonzero(solution_3d[0]):3d} | valid = {is_valid} | time = {time_taken:4.1f}s')
    if is_valid: plot_3d(solution_3d)
    else:        plot_idx(df, idx)

    assert is_valid == is_valid_solution(solution_3d[0], solution_3d[-1], delta)
    assert is_valid == is_valid_solution(solution_3d[0], board,           delta)
    assert len(solution_3d) == delta + 1

    assert is_valid == True
    assert np.all( board == solution_3d[-1] )



@pytest.mark.parametrize("game_of_life_solver", [
    game_of_life_solver,
    # game_of_life_solver_patterns,
])
@pytest.mark.parametrize("idx", [
    0,      # 2x2 square
    43612,  # pretty
])
def test_game_of_life_solver_train_df(game_of_life_solver, idx):
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

    if is_valid_solution_3d(solution_3d): plot_3d(solution_3d)
    else:                                 plot_idx(test_df, idx)

    assert is_valid_solution_3d(solution_3d), idx
    assert np.all( board == stop ), idx

