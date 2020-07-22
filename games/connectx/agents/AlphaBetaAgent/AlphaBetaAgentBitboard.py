from agents.AlphaBetaAgent.AlphaBetaAgent import AlphaBetaAgent
from core.ConnextXbitboard import ConnectXbitboard


class AlphaBetaAgentBitboard(AlphaBetaAgent):
    game_class      = ConnectXbitboard
    heuristic_class = None


# The last function defined in the file run by Kaggle in submission.csv
# BUGFIX: duplicate top-level function names in submission.py can cause a Kaggle Submission Error
def agent_AlphaBetaAgentBitboard(observation, configuration) -> int:
    return AlphaBetaAgentBitboard.agent()(observation, configuration)
