import time

import numpy as np
import z3

from constraint_satisfaction.fix_submission import is_valid_solution
from constraint_satisfaction.z3_constraints import get_exclude_solution_constraint
from constraint_satisfaction.z3_constraints import get_game_of_life_ruleset
from constraint_satisfaction.z3_constraints import get_initial_board_constraint
from constraint_satisfaction.z3_constraints import get_no_empty_boards_constraint
from constraint_satisfaction.z3_constraints import get_z3_solver
from constraint_satisfaction.z3_constraints import get_zero_point_constraint
from constraint_satisfaction.z3_solver_patterns import solver_to_numpy_3d


def game_of_life_solver(board: np.ndarray, delta=1, timeout=0, verbose=True):
    time_start = time.perf_counter()
    size       = (size_x, size_y) = board.shape

    z3_solver, t_cells = get_z3_solver(
        size=size,
        delta=delta,
    )
    z3_solver.add( get_no_empty_boards_constraint(t_cells) )
    z3_solver.add( get_initial_board_constraint(t_cells, board) )
    z3_solver.push()

    # This is a safety catch to prevent timeouts when running in Kaggle notebooks
    if timeout: z3_solver.set("timeout", int(timeout * 1000/2.5))  # timeout is in milliseconds, but inexact and ~2.5x slow

    # BUGFIX: zero_point_distance=1 breaks test_df[90081]
    # NOTE:   zero_point_distance=2 results in: 2*delta slowdown
    # NOTE:   zero_point_distance=3 results in another 2-6x slowdown (but in rare cases can be quicker)
    for t in range(1, delta+1):
        z3_solver.add( get_game_of_life_ruleset(t_cells, delta=t) )  # attempt to solve one layer at a time
        z3_solver.push()
        for zero_point_distance in [1,2]:
            z3_solver.pop()    # remove previous zero_point_constraints for current layer
            z3_solver.push()
            z3_solver.add( get_zero_point_constraint(t_cells, zero_point_distance, delta=t) )
            is_sat = (z3_solver.check() == z3.sat)
            if is_sat: break   # found a solution - skip zero_point_distance=2
        if not is_sat: break   # no solution found after zero_point_distance=2

    # Validate that forward play matches backwards solution
    solution_3d = solver_to_numpy_3d(z3_solver, t_cells)  # calls z3_solver.check()
    time_taken  = time.perf_counter() - time_start
    if np.count_nonzero(solution_3d):  # quicker than calling z3_solver.check() again
        assert is_valid_solution(solution_3d[0], board, delta)
        if verbose: print(f'game_of_life_solver() - took: {time_taken:6.1f}s | Solved! ')
    else:
        time_taken  = time.perf_counter() - time_start
        if verbose: print(f'game_of_life_solver() - took: {time_taken:6.1f}s | unsolved')
    return z3_solver, t_cells, solution_3d



def game_of_life_next_solution(z3_solver, t_cells, verbose=True):
    time_start = time.perf_counter()

    z3_solver.add( get_exclude_solution_constraint(t_cells, z3_solver) )
    solution_3d = solver_to_numpy_3d(z3_solver, t_cells)

    time_taken  = time.perf_counter() - time_start
    if verbose: print(f'game_of_life_next_solution() - took: {time_taken:.1f}s')
    return z3_solver, t_cells, solution_3d


