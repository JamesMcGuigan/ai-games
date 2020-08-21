import math
from typing import Callable
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
from fastcache import clru_cache

from core.ConnectX import ConnectX
from heuristics.BitboardGameoversHeuristic import bitboard_gameovers_heuristic
from util.vendor.cached_property import cached_property


class ConnectXBitboard(ConnectX):
    # observation   = {'mark': 1, 'board': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    # configuration = {'columns': 7, 'rows': 6, 'inarow': 4, 'steps': 1000, 'timeout': 2}

    def __init__( self, observation, configuration,
                  heuristic_class: Callable=None, heuristic_fn: Callable=None,
                  verbose=True, **kwargs ):
        super().__init__(
            observation     = observation,
            configuration   = configuration,
            heuristic_class = heuristic_class,
            heuristic_fn    = heuristic_fn,
            verbose         = verbose
        )


    ### Magic Methods

    def __hash__(self):
        return self.board  # used by @clru_cache for self

    def __eq__(self, other):
        if not isinstance(other, self.__class__): return False
        return self.board == self.board

    def __str__(self):
        return str(self.cast_numpy(self.board))


    ### Utility Methods

    # BBNN and later code uses np.array([ int64, int64 ]) format rather than int128
    @cached_property
    def bitboard(self):
        bitboard_size  = self.columns * self.rows
        played_board   = self.board & ((1 << bitboard_size) - 1)
        player_board   = self.board >> bitboard_size
        bitboard       = np.array([played_board, player_board], np.int64)
        return bitboard

    # noinspection PyMethodOverriding
    def cast_board(self, board: Union[int,List[int]]) -> int:
        """Create a bitboard representation of observation.board List[int]"""
        if isinstance(board, (int,np.int64)): return int(board)

        bitboard_size   = self.columns * self.rows
        bitboard_played = 0  # 42 bit number for if board square has been played
        bitboard_player = 0  # 42 bit number for player 0=p1 1=p2
        for n in range(bitboard_size):
            if board[n] != 0:
                bitboard_played |= (1 << n)
                if board[n] == 2:
                    bitboard_player |= (1 << n)
        bitboard = bitboard_played | bitboard_player << bitboard_size
        return bitboard

    def cast_numpy(self, board: Union[int,List[int]]) -> np.ndarray:
        """Export a numpy representation of the board for debugging"""
        output        = np.zeros((self.rows, self.columns), dtype=np.int8)
        bitboard      = self.cast_board(board)
        bitboard_size = self.columns * self.rows
        for row in range(self.rows):
            for col in range(self.columns):
                offset = col + row * self.columns
                played = bitboard & 1 << offset
                player = bitboard & 1 << offset + bitboard_size
                if played:
                    value = 2 if player else 1
                else:
                    value = 0
                output[(row,col)] = value
        return output

    @staticmethod
    @clru_cache(maxsize=None)
    def get_gameovers(rows: int, columns: int, inarow: int) -> np.array:
        """Creates a list of all winning board positions, over 4 directions: horizontal, vertical and 2 diagonals"""
        gameovers = []

        mask_horizontal  = 0
        mask_vertical    = 0
        mask_diagonal_dl = 0
        mask_diagonal_ul = 0
        for n in range(inarow):
            mask_horizontal  |= 1 << n
            mask_vertical    |= 1 << n * columns
            mask_diagonal_dl |= 1 << n * columns + n
            mask_diagonal_ul |= 1 << n * columns + (inarow - 1 - n)

        row_inner = rows    - inarow
        col_inner = columns - inarow
        for row in range(rows):
            for col in range(columns):
                offset = col + row * columns
                if col <= col_inner:
                    gameovers.append( mask_horizontal << offset )
                if row <= row_inner:
                    gameovers.append( mask_vertical << offset )
                if col <= col_inner and row <= row_inner:
                    gameovers.append( mask_diagonal_dl << offset )
                    gameovers.append( mask_diagonal_ul << offset )
        return np.array(gameovers, dtype=np.int64)


    ### Result Methods

    def get_actions(self) -> List[int]:
        # rows are counted from sky = 0; if the top row of bitboard_played is empty we can play
        # return [ action for action in depth_range(self.columns) if self.board & (1 << action) ]
        actions = []
        for action in range(self.columns):
            if not self.board & (1 << action):
                actions.append(action)
        return actions

    def result_board( self, board: Union[int,List[int]], action: int, mark: int ) -> int:
        """This returns the next observation after applying an action"""
        if not isinstance(board, int): board = self.cast_board(board)
        if board & (1 << action): return board  # if the top row of bitboard_played is full then we have an invalid move

        next_board    = board
        bitboard_size = self.columns * self.rows
        for row in range(self.rows-1, -1, -1):       # counting up from the ground
            offset = action + (row * self.columns)    # action == col
            if not (next_board & (1 << offset)):   # find the first unplayed square in bitboard_played
                next_board |= (1 << offset)
                if mark == 2:
                    next_board |= (1 << offset + bitboard_size)
                break
        return next_board

    def get_coords( self, board: int, action: int ) -> Tuple[int,int]:
        raise NotImplementedError


    ### Heuristic Methods

    @clru_cache(maxsize=None)
    def heuristic(self):
        """Delay resolution until after parentclass constructor has finished"""
        return self.heuristic_class(self) if self.heuristic_class else None

    @clru_cache(maxsize=None)
    def gameover( self ) -> bool:
        """Has the game reached a terminal game?"""
        if len( self.actions   ) == 0:        return True
        if abs( self.utility() ) == math.inf: return True
        return False

    @clru_cache(maxsize=None)
    def utility( self ) -> float:
        """ +inf for victory or -inf for loss else 0 - from the perspective of the player who made the previous move"""
        bitboard_size = self.columns * self.rows
        gameovers     = self.get_gameovers(rows=self.rows, columns=self.columns, inarow=self.inarow)
        for gameover in gameovers:
            played_tokens = self.board & gameover
            if played_tokens == gameover:                                 # do we have four cells in a row played?
                player_tokens = (self.board >> bitboard_size) & gameover  # apply mask to extract player marks
                if player_tokens == 0:         # player 1 wins
                    return math.inf if self.player_id != 1 else -math.inf  # perspective of the previous move
                if player_tokens == gameover:  # player 2 wins
                    return math.inf if self.player_id != 2 else -math.inf  # perspective of the previous move
        return 0.0  # nothing found, neither side wins


    @clru_cache(maxsize=None)
    def score( self ) -> float:
        """ For all possible connect4 gameover positions,
            check if a player has at least one token in position and that the opponent is not blocking
            return difference in score

            Winrates:
             55% vs AlphaBetaAgent - original heuristic
             70% vs AlphaBetaAgent - + math.log2(p1_can_play) % 1 == 0
             60% vs AlphaBetaAgent - + double_attack_score=1   without math.log2() (mostly draws)
             80% vs AlphaBetaAgent - + double_attack_score=1   with math.log2() (mostly wins)
             80% vs AlphaBetaAgent - + double_attack_score=2   with math.log2() (mostly wins)
             70% vs AlphaBetaAgent - + double_attack_score=4   with math.log2() (mostly wins)
             80% vs AlphaBetaAgent - + double_attack_score=8   with math.log2() (mostly wins)
            100% vs AlphaBetaAgent - + double_attack_score=0.5 with math.log2() (mostly wins)
        """
        last_player_to_move = 1 if self.player_id == 2 else 2
        if self.heuristic_class: return self.heuristic_class().score()
        if self.heuristic_fn:    return self.heuristic_fn(self.bitboard, last_player_to_move)
        else:                    return bitboard_gameovers_heuristic(self.bitboard, last_player_to_move)
