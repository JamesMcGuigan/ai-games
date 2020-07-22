from agents.AlphaBetaAgent.AlphaBetaAgent import AlphaBetaAgent
from core.ConnextXbitboard import ConnectXbitboard


class AlphaBetaAgentBitboard(AlphaBetaAgent):
    game_class      = ConnectXbitboard
    heuristic_class = None


# The last function defined in the file run by Kaggle in submission.csv
def agent(observation, configuration) -> int:
    return AlphaBetaAgentBitboard.agent()(observation, configuration)
