import functools
import math


import numba
from dataclasses import dataclass
from enum import Enum
from enum import unique
from typing import FrozenSet
from typing import List
from typing import Set
from typing import Tuple
from typing import Union

import numpy as np
from fastcache import clru_cache
from numba import njit

from core.ConnectX import ConnectX
from core.Heuristic import Heuristic
from util.vendor.cached_property import cached_property



# (1,0)  -> (-1,0)  = down -> up
# (0,1)  -> (0,-1)  = left -> right
# (1,1)  -> (-1,-1) = down+left -> up+right
# (-1,1) -> (1,-1)  = up+left   -> down+right
@unique
class Direction(Enum):
    UP_DOWN       = (1,0)
    LEFT_RIGhT    = (0,1)
    DIAGONAL_UP   = (1,1)
    DIAGONAL_DOWN = (1,-1)
Directions = frozenset( d.value for d in Direction.__members__.values() )


@functools.total_ordering
@dataclass(init=True, frozen=True)
class Line:
    game:          'ConnectX'
    cells:         FrozenSet[Tuple[int,int]]
    direction:     Direction
    mark:          int



    ### Factory Methods

    @classmethod
    @clru_cache(None)
    def line_from_position( cls, game: 'ConnectX', coord: Tuple[int, int], direction: Direction ) -> Union['Line', None]:
        # NOTE: This function doesn't improve with @jit
        mark = game.board[coord]
        if mark == 0: return None

        cells = { coord }
        for sign in [1, -1]:
            next = cls.next_coord(coord, direction, sign)
            while cls.is_valid_coord(next, game.rows, game.columns) and game.board[next] == mark:
                cells.add(next)
                next = cls.next_coord(next, direction, sign)

        return Line(
            game      = game,
            cells     = frozenset(cells),
            mark      = mark,
            direction = direction,
        )


    ### Magic Methods

    def __len__( self ):
        return len(self.cells)

    def __hash__(self):
        return hash((self.direction, self.cells))

    def __eq__(self, other):
        if not isinstance(other, self.__class__): return False
        return self.cells == other.cells and self.direction == other.direction

    def __lt__(self, other):
        return self.score < other.score

    def __repr__(self):
        args = {"mark": self.mark, "direction": self.direction, "cells": self.cells }
        return f"Line({args})"



    ### Navigation Methods

    @staticmethod
    @njit()
    def next_coord( coord: Tuple[int, int], direction: Direction, sign=1 ) -> Tuple[int,int]:
        """Use is_valid_coord to verify the coord is valid """
        return ( coord[0]+(direction[0]*sign), coord[1]+(direction[1]*sign) )

    @staticmethod
    @njit()
    def is_valid_coord( coord: Tuple[int, int], rows: int, columns: int ) -> bool:
        x,y  = coord
        if x < 0 or rows    <= x: return False
        if y < 0 or columns <= y: return False
        return True


    ### Heuristic Methods

    @cached_property
    # @njit() ### throws exceptions
    def gameover( self ) -> bool:
        return len(self) == self.game.inarow

    @cached_property
    def score( self ):
        # A line with zero liberties is dead
        # A line with two liberties is a potential double attack
        # A line of 2 with 2 liberties is worth more than a line of 3 with one liberty
        if len(self) == self.game.inarow: return math.inf
        if len(self.liberties) == 0:      return 0                         # line can't connect 4
        if len(self) + self.extension_length < self.game.inarow: return 0  # line can't connect 4
        score = ( len(self)**2 + self.extension_score ) * len(self.liberties)
        if len(self) == 1: score /= len(Directions)                                    # Discount duplicates
        return score

    @numba.generated_jit(nopython=True, nogil=True, cache=False, forceobj=False, parallel=True)
    def utility( self, player_id: int ) -> float:
        if len(self) == self.game.inarow:
            if player_id == self.mark: return  math.inf
            else:                      return -math.inf
        return 0


    @cached_property
    def extension_length( self ):
        return np.sum(list(map(len, self.extensions)))

    @cached_property
    def extension_score( self ):
        # less than 1 - ensure center col is played first
        return np.sum([ len(extension)**1.25 for extension in self.extensions ]) / ( self.game.inarow**2 )


    ### Orginal non-numba code
    @cached_property
    def liberties( self ) -> Set[Tuple[int,int]]:
        ### Numba doesn't like this syntax
        cells = {
            self.next_coord(coord, self.direction, sign)
            for coord in self.cells
            for sign in [1, -1]
        }
        cells = {
            coord
            for coord in cells
            if  self.is_valid_coord(coord, self.game.rows, self.game.columns)
                and self.game.board[coord] == 0
        }
        return cells

    # ### BUG: Numba optimized code returns zero scores
    # @cached_property
    # def liberties( self ) -> Set[Tuple[int,int]]:
    #     return self._liberties(
    #         direction=self.direction,
    #         cells=tuple(self.cells),
    #         board=self.game.board,
    #         rows=self.game.rows,
    #         columns=self.game.columns,
    #         is_valid_coord=self.is_valid_coord,
    #         next_coord=self.next_coord,
    #     )
    # @staticmethod
    # @njit()
    # def _liberties( direction, cells, board: np.ndarray, rows: int, columns: int, is_valid_coord, next_coord ) -> Set[Tuple[int,int]]:
    #     coords = set()
    #     for sign in [1, -1]:
    #         for coord in cells:
    #             next = next_coord(coord, direction, sign)
    #             coords.add(next)
    #     output = set()
    #     for coord in coords:
    #         if is_valid_coord(coord, rows, columns) and board[coord] == 0:
    #             output.add(coord)
    #     return output



    ### Orginal non-numba code
    @cached_property
    # @njit()
    def extensions( self ) -> List[FrozenSet[Tuple[int,int]]]:
        extensions = []
        for next in self.liberties:
            extension = { next }
            for sign in [1,-1]:
                while len(extension) + len(self) < self.game.inarow:
                    next = self.next_coord(next, self.direction, sign)
                    if next in self.cells:                                               break
                    if not self.is_valid_coord(next, self.game.rows, self.game.columns): break
                    if self.game.board[next] not in (0, self.mark):                      break
                    extension.add(next)
            if len(extension):
                extensions.append(frozenset(extension))
        return extensions

    # ## BUG: Numba optimized code returns zero scores
    # @cached_property
    # def extensions( self ) -> List[Set[Tuple[int,int]]]:
    #     ### tuple() seems quicker than numba.typed.List()
    #     # liberties = numba.typed.List(); [ liberties.append(l) for l    in self.liberties ]
    #     # cells     = numba.typed.List(); [ cells.append(cell)  for cell in self.cells ]
    #     return self._extensions(
    #         length_self=len(self),
    #         liberties=tuple(self.liberties),
    #         cells=tuple(self.cells),
    #         mark=self.mark,
    #         direction=self.direction,
    #         board=self.game.board,
    #         inarow=self.game.inarow,
    #         rows=self.game.rows,
    #         columns=self.game.columns,
    #         next_coord=self.next_coord,
    #         is_valid_coord=self.is_valid_coord
    #     )
    # # noinspection PySetFunctionToLiteral
    # @staticmethod
    # @njit()
    # def _extensions(length_self, liberties, cells, mark, direction, board, inarow, rows, columns, next_coord, is_valid_coord) -> List[Set[Tuple[int,int]]]:
    #     extensions = [ set([ (0,0) ]) for _ in range(0) ]  # http://numba.pydata.org/numba-doc/latest/user/troubleshoot.html#my-code-has-an-untyped-list-problem
    #     cells      = set(cells)
    #     for next in liberties:
    #         extension = set([ next ])
    #         for sign in [1,-1]:
    #             while len(extension) + length_self < inarow:
    #                 next = next_coord(next, direction, sign)
    #                 if next in cells:                            break
    #                 if not is_valid_coord(next, rows, columns):  break
    #                 if board[next] != 0 and board[next] == mark: break
    #                 extension.add(next)
    #         if len(extension):
    #             extensions.append(set(extension))
    #     return extensions



class LibertiesHeuristic(Heuristic):
    ### Heuristic Methods - relative to the current self.player_id
    ## Heuristic Methods

    cache = {}
    def __new__(cls, game: ConnectX, *args, **kwargs):
        hash = frozenset(( game.board.tobytes(), np.fliplr(game.board).tobytes() ))
        if hash not in cls.cache:
            cls.cache[hash] = object.__new__(cls)
        return cls.cache[hash]

    def __init__(self, game: ConnectX):
        super().__init__(game)
        self.game      = game
        self.board     = game.board
        self.player_id = game.player_id

    @cached_property
    def lines(self) -> List['Line']:
        lines = set()
        for (row,col) in zip(*np.where(self.board != 0)):
            if self.board[row,col] == 0: continue
            lines |= {
                Line.line_from_position(self.game, (row, col), direction)
                for direction in Directions
            }
        lines = { line for line in lines if line.score != 0 }
        return sorted(lines, reverse=True, key=len)

    @cached_property
    def gameover( self ) -> bool:
        """Has the game reached a terminal game?"""
        if len( self.game.actions ) == 0:                    return True
        if np.any([ line.gameover for line in self.lines ]): return True
        return False


    @cached_property
    def score( self ) -> float:
        """Heuristic score"""
        # mark is the next player to move - calculate score from perspective of player who just moved
        hero_score    = np.sum([ line.score for line in self.lines if line.mark != self.player_id ])
        villain_score = np.sum([ line.score for line in self.lines if line.mark == self.player_id ])
        return hero_score - villain_score

    @cached_property
    def utility(self) -> float:
        """ +inf for victory or -inf for loss else 0 - calculated from the perspective of the player who made the previous move"""
        for line in self.lines:
            if len(line) == 4:
                # mark is the next player to move - calculate score from perspective of player who just moved
                return math.inf if line.mark != self.player_id else -math.inf
            else:
                break  # self.lines is sorted by length
        return 0
