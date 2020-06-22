import functools
import math
from dataclasses import dataclass
from enum import Enum
from enum import unique
from typing import FrozenSet
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
from fastcache import clru_cache

from games.connectx.core.ConnectX import ConnectX
from games.connectx.core.Heuristic import Heuristic
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
        mark = game.board[coord]
        if mark == 0: return None

        cells = { coord }
        for sign in [1, -1]:
            next = cls.next_coord(coord, direction, sign)
            while cls._is_valid_coord(next, game) and game.board[next] == mark:
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
        return self.score() < other.score()

    def __repr__(self):
        args = {"mark": self.mark, "direction": self.direction, "cells": self.cells }
        return f"Line({args})"



    ### Navigation Methods

    @staticmethod
    def next_coord( coord: Tuple[int, int], direction: Direction, sign=1 ) -> Tuple[int,int]:
        """Use is_valid_coord to verify the coord is valid """
        return ( coord[0]+(direction[0]*sign), coord[1]+(direction[1]*sign) )

    def is_valid_coord( self, coord: Tuple[int,int] ) -> bool:
        # Profiler: 7.7% -> 6.2% runtime | avoid an extra function call within this tight loop
        game = self.game
        x,y  = coord
        if x < 0 or game.rows    <= x: return False
        if y < 0 or game.columns <= y: return False
        return True

    @staticmethod
    def _is_valid_coord( coord: Tuple[int,int], game: 'ConnectX' ) -> bool:
        x,y  = coord
        if x < 0 or game.rows    <= x: return False
        if y < 0 or game.columns <= y: return False
        return True


    ### Heuristic Methods

    @cached_property
    def gameover( self ) -> bool:
        return len(self) == self.game.inarow

    def utility( self, player_id: int ) -> float:
        if len(self) == self.game.inarow:
            if player_id == self.mark: return  math.inf
            else:                      return -math.inf
        return 0

    @cached_property
    def score( self ):
        # A line with zero liberties is dead
        # A line with two liberties is a potential double attack
        # A line of 2 with 2 liberties is worth more than a line of 3 with one liberty
        if len(self) == self.game.inarow: return math.inf
        if len(self.liberties) == 0:      return 0                                  # line can't connect 4
        if len(self) + sum(map(len, self.extensions)) < self.game.inarow: return 0  # line can't connect 4
        score = ( len(self)**2 + self.extension_score ) * len(self.liberties)
        if len(self) == 1: score /= len(Directions)                                 # Discount duplicates
        return score

    @cached_property
    def extension_score( self ):
        # less than 1 - ensure center col is played first
        return sum( len(extension)**1.25 for extension in self.extensions ) / ( self.game.inarow**2 )

    @cached_property
    def liberties( self ) -> FrozenSet[Tuple[int,int]]:
        cells = {
            self.next_coord(coord, self.direction, sign)
            for coord in self.cells
            for sign in [1, -1]
        }
        cells = {
            coord
            for coord in cells
            if  self.is_valid_coord(coord)
                and self.game.board[coord] == 0
        }
        return frozenset(cells)

    @cached_property
    def extensions( self ) -> List[FrozenSet[Tuple[int,int]]]:
        extensions = []
        for next in self.liberties:
            extension = { next }
            for sign in [1,-1]:
                while len(extension) + len(self) < self.game.inarow:
                    next = self.next_coord(next, self.direction, sign)
                    if next in self.cells:            break
                    if not self.is_valid_coord(next): break
                    if self.game.board[next] not in (0, self.mark): break
                    extension.add(next)
            if len(extension):
                extensions.append(frozenset(extension))
        return extensions



class LinesHeuristic(Heuristic):
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
        if len( self.game.actions ) == 0:               return True
        if any( line.gameover for line in self.lines ): return True
        return False


    @cached_property
    def score( self ) -> float:
        """Heuristic score"""
        # mark is the next player to move - calculate score from perspective of player who just moved
        hero_score    = sum( line.score for line in self.lines if line.mark != self.player_id )
        villain_score = sum( line.score for line in self.lines if line.mark == self.player_id )
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
