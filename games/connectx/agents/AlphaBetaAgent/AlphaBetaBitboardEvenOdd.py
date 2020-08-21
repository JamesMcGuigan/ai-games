import numpy as np

from agents.AlphaBetaAgent.AlphaBetaAgent import AlphaBetaAgent
from core.ConnextXBitboard import ConnectXBitboard
from heuristics.BitboardOddEvenHeuristic import bitboard_evenodd_heuristic


class AlphaBetaBitboardEvenOdd(AlphaBetaAgent):
    game_class      = ConnectXBitboard
    heuristic_class = None
    heuristic_fn    = bitboard_evenodd_heuristic(
        reward_gameover        = np.inf,
        reward_single_token    = 0.1,
        reward_multi           = 1,
        reward_evenodd_pair    = 4,
        reward_odd_with_player = 0.5,
        reward_odd_with_column = 1,
    )


# The last function defined in the file run by Kaggle in submission.csv
# BUGFIX: duplicate top-level function names in submission.py can cause a Kaggle Submission Error
def agent_AlphaBetaBitboardEvenOdd(observation, configuration) -> int:
    return AlphaBetaBitboardEvenOdd.agent()(observation, configuration)
