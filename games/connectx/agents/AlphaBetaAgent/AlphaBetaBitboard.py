from agents.AlphaBetaAgent.AlphaBetaAgent import AlphaBetaAgent
from core.ConnextXBitboard import ConnectXBitboard


class AlphaBetaBitboard(AlphaBetaAgent):
    game_class      = ConnectXBitboard
    heuristic_class = None
    heuristic_fn    = None


# The last function defined in the file run by Kaggle in submission.csv
# BUGFIX: duplicate top-level function names in submission.py can cause a Kaggle Submission Error
def agent_AlphaBetaBitboard(observation, configuration) -> int:
    return AlphaBetaBitboard.agent()(observation, configuration)
