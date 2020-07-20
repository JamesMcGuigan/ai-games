from copy import copy
from struct import Struct
from typing import Callable
from typing import List
from typing import Tuple
from typing import Union

import numpy as np

from core.KaggleGame import KaggleGame
from util.vendor.cached_property import cached_property



class ConnectX(KaggleGame):
    players = 2

    # observation   = {'mark': 1, 'board': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    # configuration = {'columns': 7, 'rows': 6, 'inarow': 4, 'steps': 1000, 'timeout': 2}
    def __init__( self, observation, configuration, heuristic_class: Callable=None, verbose=True, **kwargs ):
        super().__init__(observation, configuration, heuristic_class, verbose)
        self.rows:      int = configuration.rows
        self.columns:   int = configuration.columns
        self.inarow:    int = configuration.inarow
        self.timeout:   int = configuration.timeout
        self.player_id: int = observation.mark
        self.board: np.ndarray = self.cast_board(observation.board)  # Don't modify observation.board
        self.board.setflags(write=False)  # WARN: https://stackoverflow.com/questions/5541324/immutable-numpy-array#comment109695639_5541452


    ### Magic Methods

    def __hash__(self):
        return hash(self.board.tobytes())

    def __eq__(self, other):
        if not isinstance(other, self.__class__): return False
        return self.board.tobytes() == other.board.tobytes()


    ### Utility Methods

    def cast_board( self, board: Union[np.ndarray,List[int]], copy=False ) -> np.ndarray:
        if isinstance(board, np.ndarray):
            if copy: return board.copy()
            else:    return board
        else:
            board = np.array(board, dtype=np.int8).reshape(self.rows, self.columns)
            return board




    ### Result Methods

    def result( self, action ) -> 'ConnectX':
        """This returns the next KaggleGame after applying action"""
        observation = self.result_observation(self.observation, action)
        return self.__class__(observation, self.configuration, self.heuristic_class, self.verbose)

    def result_observation( self, observation: Struct, action: int ) -> Struct:
        output = copy(observation)
        output.board = self.result_board(observation.board, action, observation.mark)
        output.mark  = 2 if observation.mark == 1 else 1
        return output

    def result_board( self, board: np.ndarray, action: int, mark: int ) -> np.ndarray:
        """This returns the next observation after applying an action"""
        next_board = self.cast_board(board, copy=True)
        coords     = self.get_coords(next_board, action)
        if None not in coords:
            next_board[coords] = mark
        return next_board

    def get_coords( self, board: np.ndarray, action: int ) -> Tuple[int,int]:
        col = action if 0 <= action < self.columns else None
        row = np.count_nonzero( board[:,col] == 0 ) - 1
        if row < 0: row = None
        return (row, col)

    @cached_property
    def actions(self) -> List[int]:
        # rows are counted from sky = 0; if the top row is empty we can play
        actions = np.nonzero(self.board[0,:] == 0)[0].tolist()   # BUGFIX: Kaggle wants List[int] not np.ndarray(int64)
        return list(actions)
