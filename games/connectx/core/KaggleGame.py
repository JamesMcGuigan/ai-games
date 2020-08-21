# from __future__ import annotations

import time
from struct import Struct
from typing import Dict
from typing import List

from fastcache import clru_cache

from util.tuplize import tuplize


class KaggleGame:
    """
    This is a generic baseclass wrapper around kaggle_environments
    def agent(observation, configuration):
        game = KaggleGame(observation, configuration)
        return random.choice(game.actions())
    """

    def __init__(self, observation, configuration, heuristic_class, heuristic_fn, verbose=True):
        self.time_start         = time.perf_counter()
        self.observation        = observation
        self.configuration      = configuration
        self.verbose            = verbose
        self.player_id          = None
        self._hash              = None
        self.heuristic_class    = heuristic_class
        self.heuristic_fn       = heuristic_fn
        self.actions: List[int] = []  # self.get_actions()

    def __hash__(self):
        """Return an id for caching purposes """
        self._hash = self._hash or hash(tuplize((self.observation, self.configuration)))
        # self._hash = self._hash or self.board.tobytes()
        return self._hash


    ### Result Methods

    def get_actions( self ) -> List:
        """Return a list of valid actions"""
        raise NotImplementedError

    def result( self, action ) -> 'KaggleGame':
        """This returns the next KaggleGame after applying action"""
        observation = self.result_observation(self.observation, action)
        return self.__class__(
            observation   = observation,
            configuration = self.configuration,
            verbose       = self.verbose
        )

    def result_observation( self, observation: Struct, action ) -> Dict:
        """This returns the next observation after applying action"""
        raise  NotImplementedError
        # return copy(self.observation)



    ### Heuristic Methods

    @clru_cache(maxsize=None)
    def heuristic(self):
        """Delay resolution until after parentclass constructor has finished"""
        return self.heuristic_class(self) if self.heuristic_class else None

    @clru_cache(maxsize=None)
    def gameover( self ) -> bool:
        """Has the game reached a terminal game?"""
        if self.heuristic():
            return self.heuristic().gameover
        else:
            return len( self.actions ) == 0

    @clru_cache(maxsize=None)
    def score( self ) -> float:
        return self.heuristic().score

    @clru_cache(maxsize=None)
    def utility( self ) -> float:
        return self.heuristic().utility
