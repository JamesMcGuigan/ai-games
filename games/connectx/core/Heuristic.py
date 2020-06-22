import math

from games.connectx.core.KaggleGame import KaggleGame
from games.connectx.core.PersistentCacheAgent import PersistentCacheAgent



class Heuristic(PersistentCacheAgent):
    """ Returns heuristic_class scores relative to the self.game.player_id"""
    instances = {}
    def __new__(cls, game: KaggleGame, *args, **kwargs):
        hash = game
        if hash not in cls.instances:
            cls.instances[hash] = object.__new__(cls)
        return cls.instances[hash]

    def __init__( self, game: KaggleGame, *args, **kwargs ):
        super().__init__(*args, **kwargs)
        self.game      = game
        self.player_id = game.player_id

    def gameover( self ) -> bool:
        """Has the game reached a terminal game?"""
        return abs(self.utility(1)) == math.inf

    def score( self, player_id: int ) -> float:
        """Heuristic score"""
        raise NotImplementedError

    def utility( self, player_id: int ) -> float:
        """ +inf for victory or -inf for loss else 0 """
        raise NotImplementedError
