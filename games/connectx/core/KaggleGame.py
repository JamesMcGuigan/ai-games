from __future__ import annotations

import time
from struct import Struct
from typing import Dict
from typing import List

from util.tuplize import tuplize
from util.vendor.cached_property import cached_property



class KaggleGame:
    """
    This is a generic baseclass wrapper around kaggle_environments
    def agent(observation, configuration):
        game = KaggleGame(observation, configuration)
        return random.choice(game.actions())
    """

    def __init__(self, observation, configuration, heuristic_class, verbose=True):
        self.time_start    = time.perf_counter()
        self.observation   = observation
        self.configuration = configuration
        self.verbose       = verbose
        self.player_id     = None
        self._hash         = None
        self.heuristic_class = heuristic_class


    def __hash__(self):
        """Return an id for caching purposes """
        self._hash = self._hash or hash(tuplize((self.observation, self.configuration)))
        # self._hash = self._hash or self.board.tobytes()
        return self._hash


    ### Result Methods

    @cached_property
    def actions( self ) -> List:
        """Return a list of valid actions"""
        raise NotImplementedError

    def result( self, action ) -> KaggleGame:
        """This returns the next KaggleGame after applying action"""
        observation = self.result_observation(self.observation, action)
        return self.__class__(observation, self.configuration, self.verbose)

    def result_observation( self, observation: Struct, action ) -> Dict:
        """This returns the next observation after applying action"""
        raise  NotImplementedError
        # return copy(self.observation)



    ### Heuristic Methods

    @cached_property
    def heuristic(self):
        """Delay resolution until after parentclass constructor has finished"""
        return self.heuristic_class(self)

    @cached_property
    def gameover( self ) -> bool:
        """Has the game reached a terminal game?"""
        if self.heuristic:
            return self.heuristic.gameover
        else:
            return len( self.actions ) == 0

    @cached_property
    def score( self ) -> float:
        return self.heuristic.score

    @cached_property
    def utility( self ) -> float:
        return self.heuristic.utility
