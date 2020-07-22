import math
from typing import Callable
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
from fastcache import clru_cache
from numba import int32
from numba import int64
from numba import int8
from numba import njit
from numba import prange
from numba import typed

from core.ConnectX import ConnectX


class ConnectXBitboardNumba(ConnectX):
    # observation   = {'mark': 1, 'board': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    # configuration = {'columns': 7, 'rows': 6, 'inarow': 4, 'steps': 1000, 'timeout': 2}

    def __init__( self, observation, configuration, heuristic_class: Callable=None, verbose=True, **kwargs ):
        super().__init__(observation, configuration, heuristic_class, verbose)


    ### Magic Methods

    def __hash__(self):
        return hash(self.board)   # used by @clru_cache for self

    def __eq__(self, other):
        if not isinstance(other, self.__class__): return False
        return self.board == self.board

    def __str__(self):
        return str(self.cast_numpy(self.board))


    ### Utility Methods

    # noinspection PyMethodOverriding
    def cast_board(self, board: Union[int,List[int]]) -> Tuple[int,int]:
        """Create a bitboard representation of observation.board List[int]"""
        if isinstance(board, Tuple): return board

        bitboard_size   = self.columns * self.rows
        bitboard_played = 0  # 42 bit number for if board square has been played
        bitboard_player = 0  # 42 bit number for player 0=p1 1=p2
        for n in range(bitboard_size):
            if board[n] != 0:
                bitboard_played |= (1 << n)
                if board[n] == 2:
                    bitboard_player |= (1 << n)
        bitboard = (bitboard_played, bitboard_player)
        return bitboard

    def cast_numpy(self, board: Union[int,List[int]]) -> np.ndarray:
        """Export a numpy representation of the board for debugging"""
        output        = np.zeros((self.rows, self.columns), dtype=np.int8)
        bitboard      = self.cast_board(board)
        for row in range(self.rows):
            for col in range(self.columns):
                offset = col + row * self.columns
                played = bitboard[0] & 1 << offset
                player = bitboard[1] & 1 << offset
                if played:
                    value = 2 if player else 1
                else:
                    value = 0
                output[(row,col)] = value
        return output

    @staticmethod
    @clru_cache(maxsize=None)
    def get_gameovers(rows: int, columns: int, inarow: int) -> typed.List:
        """Creates a list of all winning board positions, over 4 directions: horizontal, vertical and 2 diagonals"""
        gameovers = typed.List()

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
        return gameovers


    ### Result Methods

    def get_actions(self) -> List[int]:
        # rows are counted from sky = 0; if the top row of bitboard_played is empty we can play
        # return [ action for action in depth_range(self.columns) if self.board & (1 << action) ]
        actions = []
        for action in range(self.columns):
            if not self.board[0] & (1 << action):
                actions.append(action)
        return actions


    def result_board( self, board: Union[Tuple[int,int],List[int]], action: int, mark: int ) -> Tuple[int,int]:
        """This returns the next observation after applying an action"""
        if not isinstance(board, Tuple): board = self.cast_board(board)
        if board[0] & (1 << action): return board  # if the top row of bitboard_played is full then we have an invalid move
        return self._result_board(board, action, mark, self.rows, self.columns)

    @staticmethod
    @clru_cache(maxsize=None)
    @njit(nogil=True, parallel=False)
    def _result_board( board: Tuple[int,int], action: int, mark: int, rows: int, columns: int ) -> Tuple[int,int]:
        next_board = list(board)
        for row in prange(rows-1, -1, -1):      # counting up from the ground
            offset = action + (row * columns)         # action == col
            if not (next_board[0] & (1 << offset)):   # find the first unplayed square in bitboard_played
                next_board[0] |= (1 << offset)
                if mark == 2:
                    next_board[1] |= (1 << offset)
                break
        return (next_board[0], next_board[1])


    def get_coords( self, board: Tuple[int,int], action: int ) -> Tuple[int,int]:
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
        # bitboard_size = self.columns * self.rows
        board     = ( int64(self.board[0]), int64(self.board[1]) )
        gameovers = self.get_gameovers(rows=self.rows, columns=self.columns, inarow=self.inarow)
        return self._utility(board, int8(self.player_id), gameovers)

    @staticmethod
    @njit(nogil=True, parallel=False)
    def _utility( board: Tuple[int,int], player_id: int, gameovers: typed.List) -> float:
        score = 0.0
        for i in prange(len(gameovers)):
            gameover      = gameovers[int32(i)]
            played_tokens = board[0] & gameover
            if played_tokens == gameover:              # do we have four cells in a row played?
                player_tokens = (board[1]) & gameover  # apply mask to extract player marks
                if player_tokens == 0:                 # player 1 wins
                    score = math.inf if player_id != 1 else -math.inf  # perspective of the previous move
                    break                              # breaks parallel=False
                if player_tokens == gameover:          # player 2 wins
                    score = math.inf if player_id != 2 else -math.inf  # perspective of the previous move
                    break                              # breaks parallel=False
        return score  # nothing found, neither side wins


    @clru_cache(maxsize=None)
    def score( self ) -> float:
        """ Returns the heuristic score from the perspective of the player who made the previous move"""
        if self.heuristic_class: return self.heuristic().score()

        board         = ( int64(self.board[0]), int64(self.board[1]) )
        bitboard_size = self.columns * self.rows
        gameovers     = self.get_gameovers(rows=self.rows, columns=self.columns, inarow=self.inarow)
        score         = self._score(board, gameovers, self.player_id, bitboard_size)
        return score

    @staticmethod
    @njit(nogil=True, parallel=False)
    def _score(board: Tuple[int,int], gameovers: typed.List, player_id: int, bitboard_size: int=6*7) -> float:
        """ For all possible connect4 gameover positions,
            check if a player has at least one token in position and that the opponent is not blocking
            return difference in score

            NOTE: for caching purposes, this function returns the score from the perspective of the p1_player

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

        invert_mask    = (1 << bitboard_size+1)-1
        player_board   = board[1]
        p1_tokens      = board[0] & (player_board ^ invert_mask)
        p2_tokens      = board[0] & player_board

        p1_score = 0.0
        p2_score = 0.0
        p1_gameovers = []
        p2_gameovers = []
        for i in prange(len(gameovers)):
            gameover    = gameovers[int8(i)]
            p1_can_play = p1_tokens & gameover
            p2_can_play = p2_tokens & gameover
            if p1_can_play and not p2_can_play:
                if   p1_can_play == gameover:       p1_score += math.inf   # Connect 4
                elif np.log2(p1_can_play) % 1 == 0: p1_score += 0.1        # Mostly ignore single square lines
                else:                               p1_score += 1; p1_gameovers.append(gameover);
            elif p2_can_play and not p1_can_play:
                if   p2_can_play == gameover:       p2_score += math.inf   # Connect 4
                elif np.log2(p2_can_play) % 1 == 0: p2_score += 0.1        # Mostly ignore single square lines
                else:                               p2_score += 1; p2_gameovers.append(gameover);

        double_attack_score = 0.5  # 0.5 == 100% winrate vs AlphaBetaAgent
        for i in prange(len(p1_gameovers)):
            gameover1 = p1_gameovers[i]
            for j in prange(i+1, len(p1_gameovers)):
                gameover2 = p1_gameovers[j]
                overlap   = gameover1 & gameover2
                if gameover1 == gameover2:      continue  # Ignore self
                if overlap   == 0:              continue  # Ignore no overlap
                if np.log2(overlap) % 1 != 0:   continue  # Only count double_attacks with a single overlap square
                p1_score += double_attack_score
        for i in prange(len(p2_gameovers)):
            gameover1 = p2_gameovers[i]
            for j in prange(i+1, len(p2_gameovers)):
                gameover2 = p2_gameovers[j]
                overlap = gameover1 & gameover2
                if gameover1 == gameover2:      continue  # Ignore self
                if overlap   == 0:              continue  # Ignore no overlap
                if np.log2(overlap) % 1 != 0:   continue  # Only count double_attacks with a single overlap square
                p2_score += double_attack_score

        if player_id != 1:
            return p1_score - p2_score
        else:
            return p2_score - p1_score
