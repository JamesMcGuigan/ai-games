import time

import numpy as np
import z3

from constraint_satisfaction.fix_submission import is_valid_solution
from constraint_satisfaction.z3_constraints import get_exclude_solution_constraint, get_game_of_life_ruleset, \
    get_initial_board_constraint, get_no_empty_boards_constraint, \
    get_t_cells, get_zero_point_constraint
from constraint_satisfaction.z3_costs import get_initial_board_accuracy
from constraint_satisfaction.z3_utils import solver_score, solver_to_numpy_3d


def game_of_life_solver(board: np.ndarray, delta: int, idx: int, timeout=0, exact=True, verbose=True):
    time_start = time.perf_counter()

    z3_solver = z3.Optimize()
    t_cells   = get_t_cells( delta=delta, size=board.shape )
    z3_solver.add( get_no_empty_boards_constraint(t_cells) )
    z3_solver.add( get_game_of_life_ruleset(t_cells) )
    if exact:
        z3_solver.add( get_initial_board_constraint(t_cells, board) )
    else:
        z3_solver.maximize(get_initial_board_accuracy(t_cells, board))
    z3_solver.push()

    # This is a safety catch to prevent timeouts when running in Kaggle notebooks
    if timeout: z3_solver.set("timeout", int(timeout * 1000/2.5))  # timeout is in milliseconds, but inexact and ~2.5x slow

    ### Adding optimization constraints can make everything really slow!
    # z3_solver.minimize( get_surface_area(t_cells, distance=2) )
    # z3_solver.minimize( get_lone_cell_count(t_cells) )
    # z3_solver.maximize( get_whitespace_area(t_cells) )
    # z3_solver.maximize( get_static_pattern_score(t_cells) )

    # cluster_history_lookup.pickle file is now now 9.3Mb zipped
    z3_result = None
    for constraint in [
        # lambda: get_static_board_constraint(t_cells, board),                      # should be quick to evaluate
        # lambda: get_repeating_board_constraint(t_cells, board, frequency=2),      # should be quick to evaluate
        # lambda: get_image_segmentation_solver_constraint(t_cells, board, delta),  # makes debugger slow + causes Kaggle memory issues
        # lambda: get_image_segmentation_csv(t_cells, idx),       # uses less memory than solver
        lambda: z3.And([]),                                       # included in get_image_segmentation_solver_constraint()
    ]:
        constraint = constraint()
        if not isinstance(constraint, z3.AstRef) and len(constraint) == 0: continue  # ignore empty constraints
        z3_solver.push()
        z3_solver.add( constraint )

        # BUGFIX: zero_point_distance=1 breaks test_df[90081]
        # NOTE:   zero_point_distance=2 results in: 2*delta slowdown
        # NOTE:   zero_point_distance=3 results in another 2-6x slowdown (but in rare cases can be quicker)
        # noinspection PyAssignmentToLoopOrWithParameter
        for constraint in [
            lambda: get_zero_point_constraint(t_cells, zero_point_distance=1),
            lambda: get_zero_point_constraint(t_cells, zero_point_distance=2),
        ]:
            z3_solver.push()
            z3_solver.add( constraint() )
            z3_result = z3_solver.check()   # can return z3.unknown when exact = False
            is_sat    = z3_result == z3.sat # or (not exact and z3_result == z3.unknown)  # z3.unknown == Fail
            if is_sat: break            # found a solution - skip zero_point_distance=2
            else:      z3_solver.pop()  # remove zero_point_constraints
        if is_sat: break                # found a solution - use it
        else:      z3_solver.pop()      # remove get_image_segmentation_solution()


    # Validate that forward play matches backwards solution
    solution_3d = solver_to_numpy_3d(z3_solver, t_cells, z3_result=z3_result)  # calls z3_solver.check()
    time_taken  = time.perf_counter() - time_start
    if np.count_nonzero(solution_3d[0]):  # quicker than calling z3_solver.check() again
        assert not exact or is_valid_solution(solution_3d[0], board, delta)
        score = solver_score( solution_3d, board, delta=delta )
        if verbose: print(f'game_of_life_solver() - took: {time_taken:6.1f}s | Solved! {100*score:5.1f}%')
    else:
        time_taken = time.perf_counter() - time_start
        if verbose: print(f'game_of_life_solver() - took: {time_taken:6.1f}s | unsolved')
    return z3_solver, t_cells, solution_3d



def game_of_life_next_solution(z3_solver, t_cells, verbose=True):
    time_start = time.perf_counter()

    z3_solver.add( get_exclude_solution_constraint(t_cells, z3_solver) )
    solution_3d = solver_to_numpy_3d(z3_solver, t_cells)

    time_taken  = time.perf_counter() - time_start
    if verbose: print(f'game_of_life_next_solution() - took: {time_taken:.1f}s')
    return z3_solver, t_cells, solution_3d


