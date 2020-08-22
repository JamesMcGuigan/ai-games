from agents.AlphaBetaAgent.AlphaBetaBitboard import AlphaBetaBitboard
from core.ConnextXBitboard import ConnectXBitboard
from heuristics.OddEvenHeuristic import oddeven_bitsquares_heuristic


class AlphaBetaOddEven(AlphaBetaBitboard):
    game_class      = ConnectXBitboard
    heuristic_class = None
    heuristic_fn    = oddeven_bitsquares_heuristic()


# The last function defined in the file run by Kaggle in submission.csv
# BUGFIX: duplicate top-level function names in submission.py can cause a Kaggle Submission Error
def agent_AlphaBetaOddEven(observation, configuration) -> int:
    return AlphaBetaOddEven.agent()(observation, configuration)
