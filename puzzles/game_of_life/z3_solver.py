import itertools
import os
import time
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
import pydash
import z3
from joblib import delayed
from joblib import Parallel

from datasets import test_df
from datasets import train_df
from game import life_step
from plot import plot_3d
from util import batch
from util import csv_to_delta
from util import csv_to_numpy
from util import numpy_to_dict


def game_of_life_ruleset(size=(25,25), delta=1, warmup=0):

    # Create a 25x25 board for each timestep we need to solve for
    # T=0 for start_time, T=delta-1 for stop_time
    max_t = warmup + delta+1
    size_x, size_y = size
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
    return z3_solver, t_cells


# The true kaggle solution requires warmup=5, but this is very slow to compute
def game_of_life_solver(board: np.ndarray, delta=1, warmup=0, verbose=True):
    time_start = time.perf_counter()

    size = (size_x, size_y) = board.shape
    z3_solver, t_cells = game_of_life_ruleset(size=size, delta=delta, warmup=warmup)

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



# Parallel(n_jobs=n_jobs)([ delayed(solve_dataframe_idx)(board, delta, idx) ])
def solve_dataframe_idx(board: np.ndarray, delta: int, idx: int, verbose=True) -> Tuple[np.ndarray,int]:
    time_start = time.perf_counter()

    z3_solver, t_cells, solution_3d = game_of_life_solver(board, delta, verbose=False)

    time_taken = time.perf_counter() - time_start
    if verbose:
        message = "Solved! " if np.count_nonzero(solution_3d) else "unsolved"
        print(f'{idx:05d}: {message} in {time_taken:.1f}s')
    return solution_3d, idx


def solve_dataframe(df: pd.DataFrame = test_df, save='submission.csv') -> pd.DataFrame:
    solved = 0
    total  = 0
    submision_df = pd.read_csv(save, index_col='id')  # manually copy/paste sample_submission.csv to location
    time_start   = time.perf_counter()

    # Create list of all remaining idxs to be solved
    original_df = df
    for delta in sorted(df['delta'].unique()):  # [1,2,3,4,5]
        # Process in assumed order of difficulty, easiest first
        df = original_df
        df = df[ df['delta'] == delta ]                               # smaller deltas are easier
        df = df.iloc[ df.apply(np.count_nonzero, axis=1).argsort() ]  # smaller grids are easier

        # Create list of unsolved idxs
        idxs = []
        for idx in df.index:
            try:
                if np.count_nonzero(submision_df.loc[idx]) != 0:
                    solved += 1
                    total  += 1
                else:
                    idxs.append(idx)
            except:
                idxs.append(idx)

        # Create multi-process batch jobs as 50,000 datapoints may take a while
        n_jobs     = os.cpu_count()
        batch_size = n_jobs * 8
        for idx_batch in batch(idxs, batch_size):
            jobs_batch = []
            for idx in idx_batch:
                delta = csv_to_delta(df, idx)
                board = csv_to_numpy(df, idx, type='stop')
                jobs_batch.append( delayed(solve_dataframe_idx)(board, delta, idx) )

            solution_idx_batch = Parallel(n_jobs=n_jobs)(jobs_batch)
            for solution_3d, idx in solution_idx_batch:
                solution_dict         = numpy_to_dict(solution_3d[0])
                submision_df.loc[idx] = pd.Series(solution_dict)

                if np.count_nonzero(solution_3d) != 0:
                    solved += 1
                total += 1

            # write to file periodically, incase of crash
            submision_df.to_csv(save)

    time_taken = time.perf_counter() - time_start
    percentage = (100 * solved / total) if total else 0
    print(f'Solved: {solved}/{total} = {percentage}% in {time_taken:.1f}s')
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