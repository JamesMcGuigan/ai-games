import itertools
import time
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
import pydash
import z3

from datasets import test_df
from util import csv_to_delta
from util import csv_to_numpy
from util import numpy_to_dict


def reverse_game_of_life_solver(board: np.ndarray, delta=1, verbose=True):
    time_start = time.perf_counter()

    # Create a 25x25 board for each timestep we need to solve for
    # T=0 for start_time, T=delta-1 for stop_time
    size_x  = board.shape[0]
    size_y  = board.shape[1]
    t_cells = [
        [
            [ z3.Int(f"{x}{y}@{t}") for y in range(size_y) ]
            for x in range(size_x)
        ]
        for t in range(0, delta+1)
    ]
    z3_solver = z3.Solver()  # create a solver instance

    # Add constraints that every box has a boolean value of 1 or 0
    z3_solver.add([ z3.Or(cell == 0, cell == 1) for cell in pydash.flatten_deep(t_cells) ])


    # Add constraints for T=delta-1 the problem defined in the input board
    z3_solver.add([
        t_cells[-1][x][y] == int(board[x][y])
        for x,y in itertools.product(range(size_x), range(size_y))
    ])


    # Rules expressed forwards:
    # living + 4-8 neighbours = dies
    # living + 2-3 neighbours = lives
    # living + 0-1 neighbour  = dies
    # dead   +   3 neighbours = lives
    for t in range(1, delta+1):
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
            current_neighbours = get_neighbourhood_cells(t_cells[t], x, y, distance=2)
            z3_solver.add([
                z3.If(
                    z3.And( cell == 0, z3.Sum(current_neighbours) == 0 ),
                    past_cell == 0,
                    True
                )
            ])

    # if z3_solver.check() != z3.sat: print('Unsolvable!')
    time_taken = time.perf_counter() - time_start
    if verbose: print(f'reverse_game_of_life_solver() - took: {time_taken:.1f}s | {z3_solver.check()}')
    return z3_solver, t_cells


def reverse_game_of_life_next_solution(z3_solver, t_cells, verbose=True):
    time_start = time.perf_counter()
    z3_solver.add(z3.Or(*[
        cell != z3_solver.model()[cell]
        for cell in pydash.flatten_deep(t_cells[0])
    ]))
    z3_solver.check()
    time_taken = time.perf_counter() - time_start
    if verbose: print(f'reverse_game_of_life_next_solution() - took: {time_taken:.1f}s')
    return z3_solver, t_cells


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


def solver_to_list_2d(z3_solver, t_cells) -> List[List[int]]:
    is_sat = z3_solver.check() == z3.sat
    output = [
        [
            int(z3_solver.model()[cell].as_string()) if is_sat else 0
            for y, cell in enumerate(cells)
        ]
        for x, cells in enumerate(t_cells[-1])
    ]
    return output

def solver_to_numpy(z3_solver, t_cells) -> np.ndarray:
    solution = solver_to_list_2d(z3_solver, t_cells)
    return np.array( solution, dtype=np.int8 )


def solve_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    solved = 0
    total  = 0
    submision_df = df.copy().drop('delta', axis=1)
    loop_start   = time.perf_counter()
    for idx, row in df.T.iteritems():
        time_start = time.perf_counter()

        delta = csv_to_delta(df, idx)
        board = csv_to_numpy(df, idx, type='stop')
        z3_solver, t_cells = reverse_game_of_life_solver(board, delta, verbose=False)  # takes ~10s
        solution_np        = solver_to_numpy(z3_solver, t_cells)
        solution_dict      = numpy_to_dict(solution_np)
        submision_df[idx]  = pd.Series(solution_dict)

        time_taken = time.perf_counter() - time_start
        total += 1
        if z3_solver.check() == z3.sat:
            solved += 1
            print(f'{idx:05d}: Solved!  in {time_taken:.1f}s')
        else:
            print(f'{idx:05d}: unsolved in {time_taken:.1f}s')

    loop_taken = time.perf_counter() - loop_start
    print(f'Solved: {solved}/{total} = {100*solved/total}% in {loop_taken:.1f}s')
    return submision_df



if __name__ == '__main__':
    # solve_dataframe(train_df).to_csv('train.csv')
    solve_dataframe(test_df).to_csv('train.csv')