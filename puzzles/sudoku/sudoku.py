#!/usr/bin/env python3

import time
from itertools import chain  # flatten nested lists; chain(*[[a, b], [c, d], ...]) == [a, b, c, d, ...]

from z3 import *


rows = 'ABCDEFGHI'
cols = '123456789'
boxes = [[Int("{}{}".format(r, c)) for c in cols] for r in rows]  # declare variables for each box in the puzzle
square_units = [ [ x+y for x in A for y in B ] for A in ('ABC','DEF','GHI') for B in ('123','456','789') ]

def sudoku_solver(board):
    s_solver = Solver()  # create a solver instance

    # TODO: Add constraints that every box has a value between 1-9 (inclusive)
    s_solver.add([ And(1 <= box, box <= 9) for box in chain(*boxes) ])

    # # TODO: Add constraints that every box in a row has a distinct value
    for i in range(len(boxes)): s_solver.add(Distinct(boxes[i]))

    # # TODO: Add constraints that every box in a column has a distinct value
    for i in range(len(boxes)): s_solver.add(Distinct([ row[i] for row in boxes ]))

    # TODO: Add constraints so that every box in a 3x3 block has a distinct value
    for rows in [[0,1,2],[3,4,5],[6,7,8]]:
        for cols in [[0,1,2],[3,4,5],[6,7,8]]:
            s_solver.add(Distinct([ boxes[r][c] for r in rows for c in cols ]))

    # TODO: Add constraints boxes[i][j] == board[i][j] for each box where board[i][j] != 0
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] != 0:
                # print(i,j,board[i][j])
                s_solver.add( boxes[i][j] == board[i][j] )

    return s_solver


import pandas as pd


### Conversion Functions

def format_time(seconds):
    if seconds < 1:     return "{:.0f}ms".format(seconds*1000)
    if seconds < 60:    return "{:.2f}s".format(seconds)
    if seconds < 60*60: return "{:.0f}m {:.0f}s".format(seconds//60, seconds%60)
    if seconds < 60*60: return "{:.0f}h {:.0f}m {:.0f}s".format(seconds//(60*60), (seconds//60)%60, seconds%60)


def string_to_tuple(string):
    if isinstance(string, Solver): string = solver_to_tuple(string)

    string = string.replace('.','0')
    output = tuple( tuple(map(int, string[n*9:n*9+9])) for n in range(0,9) )
    return output


def tuple_to_string(board, zeros='.'):
    if isinstance(board, str):    board = string_to_tuple(board)
    if isinstance(board, Solver): board = solver_to_tuple(board)

    output = "".join([ "".join(map(str,row)) for row in board ])
    output = output.replace('0', zeros)
    return output


def solver_to_tuple(s_solver):
    output = tuple(
        tuple(
            int(s_solver.model()[box].as_string())
            for col, box in enumerate(_boxes)
        )
        for row, _boxes in enumerate(boxes)
    )
    return output


def solver_to_string(s_solver, zeros='.'):
    output = "".join(
        "".join(
            s_solver.model()[box].as_string()
            for col, box in enumerate(_boxes)
        )
        for row, _boxes in enumerate(boxes)
    )
    return output


def series_to_inout_pair(series):
    input  = ''
    output = ''
    for key, value in series.iteritems():
        if isinstance(value, str) and len(value) == 9*9:
            if not input: input  = value
            else:         output = value
    return (input, output)



### Print Functions

def print_board(board):
    if isinstance(board, str):     board = string_to_tuple(board)
    if isinstance(board, Solver):  board = solver_to_tuple(board)
    for row, _boxes in enumerate(boxes):
        if row and row % 3 == 0:
            print('-'*9+"|"+'-'*9+"|"+'-'*9)
        for col, box in enumerate(_boxes):
            if col and col % 3 == 0:
                print('|', end='')
            print(' {} '.format((board[row][col] or '-')), end='')
        print()
    print()


def print_sudoku( board ):
    if isinstance(board, str): board = string_to_tuple(board)

    print_board(board)

    time_start = time.perf_counter()
    s_solver   = sudoku_solver(board)
    time_end   = time.perf_counter()
    if s_solver.check() != sat: print('Unsolvable'); return

    time_end   = time.perf_counter()
    print_board(s_solver)
    print('solved in {:.2f}s'.format(time_end - time_start))



### Solve Functions

def solve_sudoku( board, format=str ):
    """This is really just a wrapper function that deals with type conversion"""
    if isinstance(board, str):     board = string_to_tuple(board)
    if isinstance(board, Solver):  board = solver_to_tuple(board)

    s_solver = sudoku_solver(board)

    if s_solver.check() != sat:
        return None
    if format == str:
        return solver_to_string(s_solver)
    if format == tuple:
        return solver_to_tuple(s_solver)
    return s_solver


def solve_dataframe(dataframe, count=0, timeout=8*60*60, verbose=0):
    if isinstance(dataframe, str): dataframe = pd.read_csv(dataframe)

    time_start = time.perf_counter()
    total      = 0
    solved     = 0
    failed     = []
    different  = []
    for index, row in dataframe.iterrows():
        if count and index >= count:                    break
        if time.perf_counter() - time_start >= timeout: break

        board, expected = series_to_inout_pair(row)
        board  = board.replace('0', '.')
        output = solve_sudoku(board, format=str)

        if output is None:
            failed.append([index, row])
            if verbose:
                print(f"Failed:    {board} -> {expected} != {output}")
            if verbose >= 2:
                print_board(board)
                print_board('Unsolvable')
        elif output != expected:
            solved += 1
            different.append([index, row])
            if verbose:
                print(f"Different: {board} -> {expected} != {output}")
            if verbose >= 2:
                print_board(board)
                print_board(output)
        else:
            solved += 1
            if verbose:
                print(f"Solved:    {board} -> {output}")
            if verbose >= 3:
                print_board(board)
                print_board(output)
        total += 1
    time_end   = time.perf_counter()
    if verbose: print()
    time_taken = time_end-time_start
    print(f'Solved {solved}/{total} Sudoku ({len(different)} different / {len(failed)} failures) in {format_time(time_taken)} ({format_time(time_taken/total)} per sudoku)')


test_board = "..149....642.31........8........67...54...9..9....5..8...6....5.......2...5.24.81"
assert test_board == tuple_to_string(string_to_tuple(test_board))


if __name__ == '__main__':
    # use the value 0 to indicate that a box does not have an assigned value
    print('## Simple Sudoku')
    board = ((0, 0, 3, 0, 2, 0, 6, 0, 0),
             (9, 0, 0, 3, 0, 5, 0, 0, 1),
             (0, 0, 1, 8, 0, 6, 4, 0, 0),
             (0, 0, 8, 1, 0, 2, 9, 0, 0),
             (7, 0, 0, 0, 0, 0, 0, 0, 8),
             (0, 0, 6, 7, 0, 8, 2, 0, 0),
             (0, 0, 2, 6, 0, 9, 5, 0, 0),
             (8, 0, 0, 2, 0, 3, 0, 0, 9),
             (0, 0, 5, 0, 1, 0, 3, 0, 0))
    print_sudoku(board)

    print()
    print()
    print("## World's Hardest Sudoku")
    print('- https://www.telegraph.co.uk/news/science/science-news/9359579/Worlds-hardest-sudoku-can-you-crack-it.html')
    board_hardest_sudoku = (
        (8, 0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 3, 6, 0, 0, 0, 0, 0),
        (0, 7, 0, 0, 9, 0, 2, 0, 0),
        (0, 5, 0, 0, 0, 7, 0, 0, 0),
        (0, 0, 0, 0, 4, 5, 7, 0, 0),
        (0, 0, 0, 1, 0, 0, 0, 3, 0),
        (0, 0, 1, 0, 0, 0, 0, 6, 8),
        (0, 0, 8, 5, 0, 0, 0, 1, 0),
        (0, 9, 0, 0, 0, 0, 4, 0, 0)
    )
    print_sudoku(board_hardest_sudoku)
