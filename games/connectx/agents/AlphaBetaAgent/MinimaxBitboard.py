import math

from agents.AlphaBetaAgent.AlphaBetaAgent import AlphaBetaAgent
from core.ConnextXBitboard import ConnectXBitboard
from core.KaggleGame import KaggleGame


class MinimaxBitboard(AlphaBetaAgent):
    persist         = True
    game_class      = ConnectXBitboard
    heuristic_class = None

    # Always set alpha=-math.inf, beta=math.inf to disable AlphaBeta pruning - as heuristic is inadmissible
    def _alphabeta_min_value( self, game: KaggleGame, player_id, depth: int, alpha=-math.inf, beta=math.inf, endtime=0.0 ):
        return super()._alphabeta_min_value( game, player_id, depth, alpha=-math.inf, beta=math.inf, endtime=endtime )

    def _alphabeta_max_value( self, game: KaggleGame, player_id: int, depth, alpha=-math.inf, beta=math.inf, endtime=0.0  ):
        return super()._alphabeta_max_value( game, player_id, depth, alpha=-math.inf, beta=math.inf, endtime=endtime )


# The last function defined in the file run by Kaggle in submission.csv
# BUGFIX: duplicate top-level function names in submission.py can cause a Kaggle Submission Error
def agent_MinimaxBitboard(observation, configuration) -> int:
    return MinimaxBitboard.agent()(observation, configuration)
