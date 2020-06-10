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


def print_board(board):
    for row, _boxes in enumerate(boxes):
        if row and row % 3 == 0:
            print('-'*9+"|"+'-'*9+"|"+'-'*9)
        for col, box in enumerate(_boxes):
            if col and col % 3 == 0:
                print('|', end='')
            print(' {} '.format((board[row][col] or '-')), end='')
        print()
    print()


def print_sudoku_solution( board ):
    time_start = time.perf_counter()
    s_solver = sudoku_solver(board)
    assert s_solver.check() == sat, "Uh oh. The solver didn't find a solution. Check your constraints."
    for row, _boxes in enumerate(boxes):
        if row and row % 3 == 0:
            print('-'*9+"|"+'-'*9+"|"+'-'*9)
        for col, box in enumerate(_boxes):
            if col and col % 3 == 0:
                print('|', end='')
            print(' {} '.format(s_solver.model()[box]), end='')
        print()
    print()
    print('solved in', round(time.perf_counter() - time_start,2), 's')



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
    print_board(board)
    print_sudoku_solution(board)

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
    print_board(board_hardest_sudoku)
    print_sudoku_solution(board_hardest_sudoku)
