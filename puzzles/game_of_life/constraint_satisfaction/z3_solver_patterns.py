# This is another buggy implementation
# Also the current logic for detecting adding rules to detect previous patterns is incredibly slow

import itertools
import time

import numpy as np
import z3

from constraint_satisfaction.fix_submission import is_valid_solution
from constraint_satisfaction.z3_constraints import get_game_of_life_ruleset
from constraint_satisfaction.z3_constraints import get_initial_board_constraint
from constraint_satisfaction.z3_constraints import get_no_empty_boards_constraint
from constraint_satisfaction.z3_constraints import get_t_cells
from constraint_satisfaction.z3_constraints import get_zero_point_constraint
from constraint_satisfaction.z3_utils import get_neighbourhood_cells
from constraint_satisfaction.z3_utils import solver_to_numpy_3d
from hashmaps.reverse_patterns import reverse_pattern_lookup


def get_reverse_pattern_constraints(t_cells):
    ### reverse_pattern_lookup[ tuplize(current) ] = [ np.array(previous), np.array(previous) ]
    size_x, size_y = len(t_cells[0]), len(t_cells[0][0])
    constraints = []
    for t, bx, by in itertools.product(range(1, len(t_cells)), range(size_x), range(size_y)):
        print('t, bx, by', t, bx, by)
        for current_pattern, previous_patterns in reverse_pattern_lookup.items():
            current_pattern = np.array(current_pattern,  dtype=np.int8)
            if_clause    = []
            then_clauses = [ [] ] * len(previous_patterns)
            for px, py in itertools.product(range(current_pattern.shape[0]), range(current_pattern.shape[1])):
                x = ( bx + px ) % size_x
                y = ( bx + px ) % size_y
                current_cell  = t_cells[t][x][y]
                previous_cell = t_cells[t-1][x][y]

                # The pattern constraint only applies to a distance of 1 whitespace away from the alive cells
                if np.any( get_neighbourhood_cells(current_pattern, px, py) ):
                    if_clause.append( current_cell == bool(current_pattern[px][py]) )

                for i, previous_pattern in enumerate(previous_patterns):
                    if np.any( get_neighbourhood_cells(previous_pattern, px, py) ):
                        then_clauses[i].append( previous_cell == bool(previous_pattern[px][py]) )

            constraints.append(
                z3.If(
                    z3.And(if_clause),
                    z3.Or([ z3.And(then_clause) for then_clause in then_clauses ]),
                    True
                )
            )
    return constraints



# The true kaggle solution requires warmup=5, but this is very slow to compute
# noinspection PyUnboundLocalVariable
def game_of_life_solver_patterns(board: np.ndarray, delta=1, warmup=0, timeout=0, verbose=True):
    time_start = time.perf_counter()

    # BUGFIX: zero_point_distance=1 breaks test_df[90081]
    # NOTE:   zero_point_distance=2 results in: 2*delta slowdown
    # NOTE:   zero_point_distance=3 results in another 2-6x slowdown (but in rare cases can be quicker)

    z3_solver = z3.Solver()
    t_cells   = get_t_cells( size=board.shape, delta=delta )
    z3_solver.add( get_no_empty_boards_constraint(t_cells) )
    z3_solver.add( get_game_of_life_ruleset(t_cells) )
    z3_solver.add( get_initial_board_constraint(t_cells, board) )

    # This is a safety catch to prevent timeouts when running in Kaggle notebooks
    if timeout: z3_solver.set("timeout", int(timeout * 1000/2.5))  # timeout is in milliseconds, but inexact and ~2.5x slow



    z3_solver.push()
    for zero_point_distance in [1,2]:
        z3_solver.pop()
        z3_solver.push()
        z3_solver.add( get_zero_point_constraint(t_cells, zero_point_distance) )


        # if z3_solver.check() != z3.sat: print('Unsolvable!')
        solution_3d = solver_to_numpy_3d(z3_solver, t_cells[warmup:])  # calls z3_solver.check()
        time_taken  = time.perf_counter() - time_start

        # Validate that forward play matches backwards solution
        if np.count_nonzero(solution_3d):  # quicker than calling z3_solver.check() again
            assert is_valid_solution(solution_3d[0], board, delta)
            if verbose: print(f'game_of_life_solver() - took: {time_taken:6.1f}s | Solved! ')
            break
    else:
        if verbose: print(f'game_of_life_solver() - took: {time_taken:6.1f}s | unsolved')
    return z3_solver, t_cells, solution_3d


