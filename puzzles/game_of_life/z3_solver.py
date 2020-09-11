import itertools
import time
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
import pydash
import z3

from datasets import train_df
from game import life_step
from plot import plot_3d
from util import csv_to_delta
from util import csv_to_numpy
from util import numpy_to_dict


# The true kaggle solution requires warmup=5, but this is very slow to compute
def game_of_life_solver(board: np.ndarray, delta=1, warmup=0, verbose=True):
    time_start = time.perf_counter()
    max_t      = warmup + delta+1

    # Create a 25x25 board for each timestep we need to solve for
    # T=0 for start_time, T=delta-1 for stop_time
    size_x  = board.shape[0]
    size_y  = board.shape[1]
    t_cells = [
        [
            [ z3.Int(f"({x:02d},{y:02d})@t={t}") for y in range(size_y) ]
            for x in range(size_x)
        ]
        for t in range(0, max_t+1)
    ]
    z3_solver = z3.Solver()  # create a solver instance

    # Add constraints that every box has a boolean value of 1 or 0
    z3_solver.add([ z3.Or(cell == 0, cell == 1) for cell in pydash.flatten_deep(t_cells) ])

    # Rules expressed forwards:
    # living + 4-8 neighbours = dies
    # living + 2-3 neighbours = lives
    # living + 0-1 neighbour  = dies
    # dead   +   3 neighbours = lives
    for t in range(1, max_t+1):
        for x,y in itertools.product(range(size_x), range(size_y)):
            cell               = t_cells[t][x][y]
            past_cell          = t_cells[t-1][x][y]
            past_neighbours    = get_neighbourhood_cells(t_cells[t-1], x, y)  # excludes self
            z3_solver.add([
                cell == z3.If(
                    z3.Or(
                        # dead + 3 neighbours = lives
                        z3.And(
                            past_cell == 0,
                            z3.Sum( *past_neighbours ) == 3,
                        ),
                        # living + 2-3 neighbours = lives
                        z3.And(
                            past_cell == 1,
                            z3.Or(
                                z3.Sum( *past_neighbours ) == 2,
                                z3.Sum( *past_neighbours ) == 3,
                            )
                        )
                    ),
                    1, 0  # cast back from boolean to int
                )
            ])

            # Ignore any currently dead cell with 0 neighbours,
            # This considerably reduces the state space and prevents zero-point energy solutions
            current_neighbours = get_neighbourhood_cells(t_cells[t], x, y, distance=1)
            z3_solver.add([
                z3.If(
                    z3.And( cell == 0, z3.Sum( *current_neighbours ) == 0 ),
                    past_cell == 0,
                    True
                )
            ])

    z3_solver.push()  # Create checkpoint before dataset constraints

    # Add constraints for T=delta-1 the problem defined in the input board
    z3_solver.add([
        t_cells[-1][x][y] == int(board[x][y])
        for x,y in itertools.product(range(size_x), range(size_y))
    ])


    # if z3_solver.check() != z3.sat: print('Unsolvable!')
    solution_3d = solver_to_numpy_3d(z3_solver, t_cells[warmup:])  # calls z3_solver.check()

    # Validate that forward play matches backwards solution
    if np.count_nonzero(solution_3d):  # quicker than calling z3_solver.check() again
        for t in range(0, delta):
            assert np.all( life_step(solution_3d[t]) == solution_3d[t+1] )

    time_taken  = time.perf_counter() - time_start
    if verbose: print(f'game_of_life_solver() - took: {time_taken:.1f}s | {"Solved! " if np.count_nonzero(solution_3d) else "unsolved" }')
    return z3_solver, t_cells, solution_3d


def game_of_life_next_solution(z3_solver, t_cells, verbose=True):
    time_start = time.perf_counter()
    z3_solver.add(z3.Or(*[
        cell != z3_solver.model()[cell]
        for cell in pydash.flatten_deep(t_cells[0])
    ]))
    solution_3d = solver_to_numpy_3d(z3_solver, t_cells)
    time_taken  = time.perf_counter() - time_start
    if verbose: print(f'game_of_life_next_solution() - took: {time_taken:.1f}s')
    return z3_solver, t_cells, solution_3d


def get_neighbourhood_coords(board: List[List[int]], x: int, y: int, distance=1) -> List[Tuple[int,int]]:
    output = []
    for dx, dy in itertools.product(range(-distance,distance+1), range(-distance,distance+1)):
        if dx == dy == 0: continue      # ignore self
        nx = (x + dx) % len(board)      # wrap board
        ny = (y + dy) % len(board[0])
        output.append( (nx, ny) )
    return output


def get_neighbourhood_cells(cells: List[List[int]], x: int, y: int, distance=1) -> List[int]:
    coords = get_neighbourhood_coords(cells, x, y, distance)
    output = [ cells[x][y] for (x,y) in coords ]
    return output


def solver_to_numpy_3d(z3_solver, t_cells) -> np.ndarray:  # np.int8[time][x][y]
    is_sat = z3_solver.check() == z3.sat
    output = np.array([
        [
            [
                int(z3_solver.model()[cell].as_string()) if is_sat else 0
                for y, cell in enumerate(cells)
            ]
            for x, cells in enumerate(t_cells[t])
        ]
        for t in range(len(t_cells))
    ], dtype=np.int8)
    return output


def solve_dataframe(df: pd.DataFrame, save='submission.csv') -> pd.DataFrame:
    solved = 0
    total  = 0
    submision_df = df.copy().drop('delta', axis=1)
    loop_start   = time.perf_counter()
    for idx, row in df.T.iteritems():
        time_start = time.perf_counter()

        delta = csv_to_delta(df, idx)
        board = csv_to_numpy(df, idx, type='stop')
        z3_solver, t_cells, solution_3d = game_of_life_solver(board, delta, verbose=False)  # takes ~10s
        solution_dict     = numpy_to_dict(solution_3d[0])
        submision_df[idx] = pd.Series(solution_dict)

        time_taken = time.perf_counter() - time_start
        total += 1
        if z3_solver.check() == z3.sat:
            solved += 1
            print(f'{idx:05d}: Solved!  in {time_taken:.1f}s')
        else:
            print(f'{idx:05d}: unsolved in {time_taken:.1f}s')

        if save:
            submision_df.to_csv(save)

    loop_taken = time.perf_counter() - loop_start
    print(f'Solved: {solved}/{total} = {100*solved/total}% in {loop_taken:.1f}s')
    return submision_df



if __name__ == '__main__':
    for df, idx in [
        (train_df, 0),  # delta = 3 - solved  9.8s
        (train_df, 4),  # delta = 1 - solved  3.9s
        (train_df, 0),  # delta = 3
    ]:
        delta    = csv_to_delta(df, idx)
        board    = csv_to_numpy(df, idx, type='stop')
        expected = csv_to_numpy(df, idx, type='start')
        z3_solver, t_cells, solution_3d = game_of_life_solver(board, delta)
        plot_3d(solution_3d)

    # solve_dataframe(train_df, save=None)
    # solve_dataframe(test_df, save='submission.csv')