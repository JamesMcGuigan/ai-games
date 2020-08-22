from agents.AlphaBetaAgent.AlphaBetaBitboard import AlphaBetaBitboard
from core.ConnextXBitboard import ConnectXBitboard
from heuristics.BitsquaresHeuristic import bitsquares_heuristic


class AlphaBetaBitsquares(AlphaBetaBitboard):
    game_class      = ConnectXBitboard
    heuristic_class = None
    heuristic_fn    = bitsquares_heuristic()


# The last function defined in the file run by Kaggle in submission.csv
# BUGFIX: duplicate top-level function names in submission.py can cause a Kaggle Submission Error
def agent_AlphaBetaBitsquares(observation, configuration) -> int:
    return AlphaBetaBitsquares.agent()(observation, configuration)
