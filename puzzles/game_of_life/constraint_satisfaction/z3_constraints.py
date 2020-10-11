import itertools

import numpy as np
import pydash
import z3

from constraint_satisfaction.z3_utils import get_neighbourhood_cells
from image_segmentation.image_segmentation_solver import image_segmentation_solver  # expensive filesystem import
from utils.game import life_step_3d
from utils.util import csv_to_numpy


def get_t_cells(delta=1, size=(25,25)):
    t_cells = [
        [
            [ z3.Bool(f"({x:02d},{y:02d})@t={t}") for y in range(size[1]) ]
            for x in range(size[0])
        ]
        for t in range(0, delta+1)
    ]
    return t_cells



def get_game_of_life_ruleset(t_cells, delta=0):
    # Create a 25x25 board for each timestep we need to solve for
    # T=0 for start_time, T=delta-1 for stop_time
    size_x, size_y = len(t_cells[0]), len(t_cells[0][0])
    delta = delta or len(t_cells)-1

    # Rules expressed forwards:
    # living + 4-8 neighbours = dies
    # living + 2-3 neighbours = lives
    # living + 0-1 neighbour  = dies
    # dead   +   3 neighbours = lives
    constraints = []
    for t in range(len(t_cells)-delta, len(t_cells)):
        for x,y in itertools.product(range(size_x), range(size_y)):
            cell            = t_cells[t][x][y]
            past_cell       = t_cells[t-1][x][y]
            past_neighbours = get_neighbourhood_cells(t_cells[t-1], x, y)  # excludes self
            constraints.append(
                # dead   + 3 neighbours   = lives
                # living + 2-3 neighbours = lives
                cell == z3.And([
                    z3.AtLeast( past_cell, *past_neighbours, 3 ),
                    z3.AtMost(             *past_neighbours, 3 ),
                ])
            )
    return constraints



def get_no_empty_boards_constraint(t_cells):
    """ Add constraint that there can be no empty boards """
    constraints = []
    for t in range(0, len(t_cells)-1):
        layer = pydash.flatten_deep(t_cells[t])
        constraints.append( z3.AtLeast( *layer, 3 ) )  # Three is a minimum viable board
    return constraints



def get_zero_point_constraint(t_cells, zero_point_distance: int, delta=0):
    # Ignore any currently dead cell with 0 neighbours,
    # This considerably reduces the state space and prevents zero-point energy solutions
    # BUGFIX: distance=1 breaks test_df[90081]
    size_x, size_y = ( len(t_cells[0]), len(t_cells[0][0]) )
    delta = delta or len(t_cells)-1
    constraints = []
    for t in range(len(t_cells)-delta, len(t_cells)):
        for x,y in itertools.product(range(size_x), range(size_y)):
            cell      = t_cells[t][x][y]
            past_cell = t_cells[t-1][x][y]
            if zero_point_distance:
                current_neighbours = get_neighbourhood_cells(t_cells[t], x, y, distance=zero_point_distance)
                constraints.append(
                    z3.If(
                        z3.AtMost( cell, *current_neighbours, 0 ),
                        past_cell == False,
                        True
                    )
                )
    return constraints



def get_initial_board_constraint(t_cells, board, delta=0):
    """ assert all( t_cells[-1] == board ) """
    constraints = [
        t_cells[-1-delta][x][y] == bool(board[x][y])
        for x,y in itertools.product(range(board.shape[0]), range(board.shape[1]))
    ]
    return constraints


def get_static_board_constraint(t_cells, board):
    constraints = [
        t_cells[t][x][y] == bool(board[x][y])
        for t in range(len(t_cells)-1)
        for x,y in itertools.product(range(board.shape[0]), range(board.shape[1]))
    ]
    return constraints


def get_repeating_board_constraint(t_cells, board, frequency=2):
    if frequency <= len(t_cells)-1: frequency = 1   # ensure we don't constrain to a blank board
    constraints = [
        t_cells[t][x][y] == bool(board[x][y])
        for t in range(len(t_cells)-1)
        for x,y in itertools.product(range(board.shape[0]), range(board.shape[1]))
        if len(t_cells)-t % frequency == 0
    ]
    return constraints



def get_exclude_solution_constraint(t_cells, z3_solver):
    """ assert any( t_cells[0] != z3_solver[0] ) """
    return z3.Or(*[
        cell != z3_solver.model()[cell]
        for cell in pydash.flatten_deep(t_cells[0])
    ])


def get_image_segmentation_solver_constraint(t_cells, stop_board, delta):
    constraints  = []
    start_boards = image_segmentation_solver(stop_board, delta)
    for start_board in start_boards:
        if np.count_nonzero(start_board) == 0: continue  # ignore empty boards
        timeline = life_step_3d(start_board, delta)
        solution = [
            t_cells[t][x][y] == bool(timeline[t][x][y])
            for t in range(len(timeline))
            for x,y in itertools.product(range(start_board.shape[0]), range(start_board.shape[1]))
            if bool(timeline[t][x][y])  # only set constraint for alive cells
        ]
        if len(solution): constraints.append( z3.And(solution) )
    return z3.Or( constraints ) if len(constraints) else []  # return empty list if nothing is found


def get_image_segmentation_csv(t_cells, idx):
    from utils.datasets import image_segmentation_df  # expensive filesystem import

    if idx in image_segmentation_df.index:
        start_board = csv_to_numpy( image_segmentation_df, idx, key='start' )
        constraints = [
            t_cells[0][x][y] == bool(start_board[x][y])
            for x,y in itertools.product(range(start_board.shape[0]), range(start_board.shape[1]))
            if bool(start_board[x][y])  # only set constraint for alive cells
        ]
        return constraints
    else:
        return []
