import itertools
import time

import numpy as np
import pydash
import z3

from constraint_satisfaction.fix_submission import is_valid_solution
from constraint_satisfaction.z3_constraints import get_neighbourhood_cells
from constraint_satisfaction.z3_solver import get_zero_point_constraint
from constraint_satisfaction.z3_utils import solver_to_numpy_3d


def add_cell_count_minimization(z3_solver, t_cells, delta):
    cost = z3.Sum([
        z3.If(cell,1,0)
        for cell in pydash.flatten_deep(t_cells[-delta-1:-1])
    ])
    z3_solver.minimize(cost)
    return z3_solver


def game_of_life_solver_iterative_delta(board: np.ndarray, delta=1, timeout=0, max_backtracks=10, minimize=False, verbose=True):
    time_start = time.perf_counter()
    size       = (size_x, size_y) = board.shape

    # Create an initial solver
    z3_solver = z3.Optimize()
    z3_solver, t_cells = game_of_life_ruleset_iterative(size=size, delta=0, max_t=delta, z3_solver=z3_solver)

    # Add constraints for T=delta-1 the problem defined in the input board
    z3_solver.add([
        t_cells[-1][x][y] == bool(board[x][y])
        for x,y in itertools.product(range(size_x), range(size_y))
    ])
    z3_solver.push()

    # This is a safety catch to prevent timeouts when running in Kaggle notebooks
    if timeout: z3_solver.set("timeout", int(timeout * 1000/2.5))  # timeout is in milliseconds, but inexact and ~2.5x slow

    solution_3d     = np.zeros((delta+1, size_x, size_y), dtype=np.int8)
    backtrack_count = 0

    # this is a fancy way to write a for loop, which allows us to repeat and insert deltas if failed
    deltas = list(range(1, delta+1))
    while len(deltas):
        t = deltas[0]
        if max_backtracks and max_backtracks < backtrack_count: break  # exit quickly if we have an unsat board

        z3_solver, t_cells = game_of_life_ruleset_iterative(size=size, delta=t, max_t=delta, z3_solver=z3_solver, t_cells=t_cells)
        if minimize: add_cell_count_minimization(z3_solver, t_cells, t)

        z3_solver.push()       # save ruleset constraints
        for zero_point_distance in [1,2]:
            z3_solver.pop()    # remove previous zero point constraints
            z3_solver.push()   # we need a push for every pop
            z3_solver.add( get_zero_point_constraint(t_cells, zero_point_distance) )
            is_sat = z3_solver.check()

            if is_sat == z3.sat:
                solution_3d = solver_to_numpy_3d(z3_solver, t_cells)  # calls z3_solver.check()
                if is_valid_solution(solution_3d[delta-t], board, t):
                    # Fix this solution as a constraint
                    z3_solver.add(z3.And(*[
                        t_cells[delta-t][x][y] == bool( solution_3d[delta-t][x][y] )
                        for x in range(size_x)
                        for y in range(size_y)
                    ]))
                    deltas.remove(t)  # move to next iteration of the whilefor loop
                    break             # exit zero_point_distance for loop
        else:
            # We couldn't find a solution, backup and find a different solution for the previous delta
            z3_solver.pop()    # remove previous ruleset constraints
            if t == 1: break   # unsat at top layers -> try zero_point_distance=2
            z3_solver.add(z3.Or(*[
                t_cells[delta-(t-1)][x][y] != bool( solution_3d[delta-(t-1)][x][y] )
                for x in range(size_x)
                for y in range(size_y)
            ]))
            deltas.insert(0, t-1)  # reattempt the previous layer
            backtrack_count += 1
            # print('backtrack_count', backtrack_count)

    time_taken  = time.perf_counter() - time_start

    # Validate that forward play matches backwards solution
    if is_valid_solution(solution_3d[0], board, delta):  # quicker than calling z3_solver.check() again
        if verbose: print(f'game_of_life_solver() - took: {time_taken:6.1f}s | Solved! ')
    else:
        if verbose: print(f'game_of_life_solver() - took: {time_taken:6.1f}s | unsolved')
    return z3_solver, t_cells, solution_3d



def game_of_life_ruleset_iterative(size=(25, 25), delta=1, max_t=0, z3_solver=None, t_cells=None):
    """
    This is a modified and potentially buggy version of game_of_life_ruleset()
    """
    max_t = max_t or delta  # BUGFIX: delta not delta-1 | see fix_submission()
    size_x, size_y = size
    t_cells = [
        [
            [ z3.Bool(f"({x:02d},{y:02d})@t={t}") for y in range(size_y) ]
            for x in range(size_x)
        ]
        for t in range(0, max_t+1)
    ] if t_cells is None else t_cells
    z3_solver = z3.Solver() if z3_solver is None else z3_solver  # create a solver instance

    # Rules expressed forwards:
    # living + 4-8 neighbours = dies
    # living + 2-3 neighbours = lives
    # living + 0-1 neighbour  = dies
    # dead   +   3 neighbours = lives
    for t in range(max_t, max_t-delta, -1):  # loop backwards in time as max_t not always equals delta
        assert t >= 1
        for x,y in itertools.product(range(size_x), range(size_y)):
            cell            = t_cells[t][x][y]
            past_cell       = t_cells[t-1][x][y]
            past_neighbours = get_neighbourhood_cells(t_cells[t-1], x, y)  # excludes self
            z3_solver.add([
                # dead   + 3 neighbours   = lives
                # living + 2-3 neighbours = lives
                cell == z3.And([
                    z3.AtLeast( past_cell, *past_neighbours, 3 ),
                    z3.AtMost(             *past_neighbours, 3 ),
                ])
            ])

    # Add constraint that there can be no empty boards
    # At least three cells in every board, as that is the minimum required to persist
    # Don't add constraint to input t_cells[delta] layer
    for t in range(0, max_t):
        layer = pydash.flatten_deep(t_cells[t])
        z3_solver.add([ z3.AtLeast( *layer, 3 ) ])

    z3_solver.push()  # Create checkpoint before dataset constraints
    return z3_solver, t_cells
