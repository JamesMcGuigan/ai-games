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

    def __init__(self, observation, configuration, verbose=True):
        self.time_start    = time.perf_counter()
        self.observation   = observation
        self.configuration = configuration
        self.verbose       = verbose

    def __hash__(self):
        """Return an id for caching purposes """
        return hash(tuplize((self.observation, self.configuration)))
        # return self.board.tobytes()


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
    def gameover( self ) -> bool:
        """Has the game reached a terminal game?"""
        raise NotImplementedError

    def score( self, player_id: int ) -> bool:
        raise NotImplementedError

    def utility( self, player_id: int ) -> bool:
        raise NotImplementedError
