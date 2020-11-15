### Optimization Cost Functions
### These
import itertools

import pydash
import z3

from constraint_satisfaction.z3_utils import get_neighbourhood_cells


def get_initial_board_accuracy(t_cells, board, delta=0):
    """ Reward = +1 for each correct cell """
    reward = z3.Sum([
        z3.If(t_cells[-1-delta][x][y] == bool(board[x][y]), 1, 0 )
        for x,y in itertools.product(range(board.shape[0]), range(board.shape[1]))
    ])
    return reward


def get_cell_count(t_cells):
    return z3.Sum([
        z3.If(cell,1,0)
        for t in range(len(t_cells))
        for cell in pydash.flatten_deep(t_cells[t])
    ])


def get_lone_cell_count(t_cells):
    """
    Empty Cell cost of 0   can result in attempting to fill the board with lots of extra squares, to avoid lone squares
    Empty Cell cost of 0.5 is somewhat performant of small boards, but still gets stuck forever on some boards
    Empty Cell cost of 2   is very slow
    """
    return z3.Sum([
        z3.If(
            t_cells[t][x][y],
            z3.If(
                z3.AtMost( *get_neighbourhood_cells(t_cells[t], x, y), 0 ),
                1,
                0
            ),
            0.5,  # cost of empty cell
        )
        for t in range(len(t_cells))
        for x in range(len(t_cells[0]))
        for y in range(len(t_cells[0][0]))
    ])


def get_whitespace_area(t_cells):
    return z3.Sum([
        z3.If(
            z3.AtMost( t_cells[t][x][y], *get_neighbourhood_cells(t_cells[t], x, y, distance=1), 0 ),
            1,
            0
        )
        for t in range(len(t_cells))
        for x in range(len(t_cells[0]))
        for y in range(len(t_cells[0][0]))
    ])


def get_static_pattern_score(t_cells, distance=1):
    """ This is incredibly slow """
    return z3.Sum([
        z3.If(
            z3.Or([
                z3.AtMost(*[
                    cell1 != cell2
                    for cell1, cell2 in zip(
                        get_neighbourhood_cells(t_cells[-1], x, y, distance=distance),
                        get_neighbourhood_cells(t_cells[t],  x, y, distance=distance)
                    )
                ], 0)
                for t in range(len(t_cells)-1)
            ]),
            1,
            0,
        )
        for x in range(len(t_cells[0]))
        for y in range(len(t_cells[0][0]))
    ])
