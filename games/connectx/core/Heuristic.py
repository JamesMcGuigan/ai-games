import math

from games.connectx.core.KaggleGame import KaggleGame
from games.connectx.core.PersistentCacheAgent import PersistentCacheAgent
from util.vendor.cached_property import cached_property



class Heuristic(PersistentCacheAgent):
    """ Returns heuristic_class scores relative to the self.game.player_id"""
    cache = {}
    def __new__(cls, game: KaggleGame, *args, **kwargs):
        hash = game
        if hash not in cls.cache:
            cls.cache[hash] = object.__new__(cls)
        return cls.cache[hash]

    def __init__( self, game: KaggleGame, *args, **kwargs ):
        super().__init__(*args, **kwargs)
        self.game      = game
        self.player_id = game.player_id

    @cached_property
    def gameover( self ) -> bool:
        """Has the game reached a terminal game?"""
        return abs(self.utility) == math.inf

    @cached_property
    def score( self ) -> float:
        """Heuristic score"""
        raise NotImplementedError

    @cached_property
    def utility(self ) -> float:
        """ +inf for victory or -inf for loss else 0 """
        raise NotImplementedError
