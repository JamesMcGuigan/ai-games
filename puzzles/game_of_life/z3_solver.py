import itertools
import os
import random
import time
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
import pydash
import z3
from pathos.multiprocessing import ProcessPool

from datasets import test_df
from datasets import train_df
from game import life_step
from plot import plot_3d
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
            [ z3.Bool(f"({x:02d},{y:02d})@t={t}") for y in range(size_y) ]
            for x in range(size_x)
        ]
        for t in range(0, max_t+1)
    ]
    z3_solver = z3.Solver()  # create a solver instance

    # Rules expressed forwards:
    # living + 4-8 neighbours = dies
    # living + 2-3 neighbours = lives
    # living + 0-1 neighbour  = dies
    # dead   +   3 neighbours = lives
    for t in range(1, max_t+1):
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

            # Ignore any currently dead cell with 0 neighbours,
            # This considerably reduces the state space and prevents zero-point energy solutions
            current_neighbours = get_neighbourhood_cells(t_cells[t], x, y, distance=1)
            z3_solver.add([
                z3.If(
                    z3.AtMost( cell, *current_neighbours, 0 ),
                    past_cell == False,
                    True
                )
            ])

    # Add constraint that there can be no empty boards
    for t in range(1, max_t+1):
        layer = pydash.flatten_deep(t_cells[t])
        z3_solver.add([ z3.AtLeast( *layer, 1 ) ])

    z3_solver.push()  # Create checkpoint before dataset constraints
    return z3_solver, t_cells


# The true kaggle solution requires warmup=5, but this is very slow to compute
def game_of_life_solver(board: np.ndarray, delta=1, warmup=0, verbose=True):
    time_start = time.perf_counter()

    size = (size_x, size_y) = board.shape
    z3_solver, t_cells = game_of_life_ruleset(size=size, delta=delta, warmup=warmup)

    # Add constraints for T=delta-1 the problem defined in the input board
    z3_solver.add([
        t_cells[-1][x][y] == bool(board[x][y])
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
                int(z3.is_true(z3_solver.model()[cell])) if is_sat else 0
                for y, cell in enumerate(cells)
            ]
            for x, cells in enumerate(t_cells[t])
        ]
        for t in range(len(t_cells))
    ], dtype=np.int8)
    return output



def get_unsolved_idxs(df: pd.DataFrame, submision_df: pd.DataFrame, sort_cells=True, sort_delta=False) -> List[int]:
    idxs = []
    deltas = sorted(df['delta'].unique()) if sort_delta else [0]
    for delta in deltas:  # [1,2,3,4,5]
        # Process in assumed order of difficulty, easiest first
        if sort_delta:
            df = df[ df['delta'] == delta ]                               # smaller deltas are easier
        if sort_cells:
            df = df.iloc[ df.apply(np.count_nonzero, axis=1).argsort() ]  # smaller grids are easier

        # Create list of unsolved idxs
        for idx in df.index:
            try:
                if np.count_nonzero(submision_df.loc[idx]) == 0:
                    idxs.append(idx)
            except:
                idxs.append(idx)

    if sort_cells == 'random':
        random.shuffle(idxs)
    return idxs


# Parallel(n_jobs=n_jobs)([ delayed(solve_dataframe_idx)(board, delta, idx) ])
def solve_dataframe_idx(board: np.ndarray, delta: int, idx: int, verbose=True) -> Tuple[np.ndarray,int]:
    time_start = time.perf_counter()

    z3_solver, t_cells, solution_3d = game_of_life_solver(board, delta, verbose=False)

    time_taken = time.perf_counter() - time_start
    if verbose:
        message = "Solved! " if np.count_nonzero(solution_3d) else "unsolved"
        print(f'{idx:05d} | delta = {delta} | cells = {np.count_nonzero(board):3d} -> {np.count_nonzero(solution_3d):3d} | {message} in {time_taken:.1f}s')
    return solution_3d, idx


def solve_dataframe(
        df: pd.DataFrame = test_df,
        save='submission.csv',
        timeout=0,
        max_count=0,
        sort_cells=True,
        sort_delta=True,
) -> pd.DataFrame:
    time_start = time.perf_counter()

    submision_df = pd.read_csv(save, index_col='id')  # manually copy/paste sample_submission.csv to location
    solved = 0
    total  = 0  # only count number solved in current runtime, ignore history

    # Pathos multiprocessing allows iterator semantics, whilst joblib has reduced CPU usage at the end of each batche
    cpus = os.cpu_count() * 3//4  # 75% CPU load to optimize for CPU cache
    pool = ProcessPool(ncpus=cpus)
    try:
        idxs   = get_unsolved_idxs(df, submision_df, sort_cells=sort_cells, sort_delta=sort_delta)
        deltas = ( csv_to_delta(df, idx)             for idx in idxs )  # generator
        boards = ( csv_to_numpy(df, idx, key='stop') for idx in idxs )  # generator

        solution_idx_iter = pool.uimap(solve_dataframe_idx, boards, deltas, idxs)
        for solution_3d, idx in solution_idx_iter:
            total += 1
            if np.count_nonzero(solution_3d) != 0:
                solved += 1

                # Reread file, update and persist
                submision_df          = pd.read_csv(save, index_col='id')
                solution_dict         = numpy_to_dict(solution_3d[0])
                submision_df.loc[idx] = pd.Series(solution_dict)
                submision_df.to_csv(save)

            # timeouts for kaggle submissions
            if max_count and max_count <= total:                            raise TimeoutError()
            if timeout   and timeout   <= time.perf_counter() - time_start: raise TimeoutError()
    except: pass
    finally:
        time_taken = time.perf_counter() - time_start
        percentage = (100 * solved / total) if total else 0
        print(f'Solved: {solved}/{total} = {percentage}% in {time_taken:.1f}s')

        pool.terminate()
        pool.clear()
    return submision_df



if __name__ == '__main__':
    # With boolean logic gives 2-4x speedup over integer logic
    for df, idx in [
        (train_df, 0),  # delta = 3 - solved  4.2s (boolean logic) | 16.1s (integer logic)
        (train_df, 4),  # delta = 1 - solved  7.6s (boolean logic) | 57.6s (integer logic)
        (train_df, 0),  # delta = 3 - solved  4.0s (boolean logic) | 17.3s (integer logic)
    ]:
        delta    = csv_to_delta(df, idx)
        board    = csv_to_numpy(df, idx, key='stop')
        expected = csv_to_numpy(df, idx, key='start')
        z3_solver, t_cells, solution_3d = game_of_life_solver(board, delta)
        plot_3d(solution_3d)

    # solve_dataframe(train_df, save=None)
    # solve_dataframe(test_df, save='submission.csv')
