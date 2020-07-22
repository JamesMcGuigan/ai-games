from agents.AlphaBetaAgent.AlphaBetaAgent import AlphaBetaAgent
from core.ConnextXBitboardNumba import ConnectXBitboardNumba


class AlphaBetaBitboardNumba(AlphaBetaAgent):
    game_class      = ConnectXBitboardNumba
    heuristic_class = None


# The last function defined in the file run by Kaggle in submission.csv
# BUGFIX: duplicate top-level function names in submission.py can cause a Kaggle Submission Error
def agent_AlphaBetaBitboardNumba(observation, configuration) -> int:
    return AlphaBetaBitboardNumba.agent()(observation, configuration)
