from games.connectx.agents.AlphaBetaAgent import AlphaBetaAgent



class AlphaBetaAgentOdd(AlphaBetaAgent):
    defaults = {
        **AlphaBetaAgent.defaults,
        "search_step":  2    # 2 to only evalue odd numbers of depths
    }

# The last function defined in the file run by Kaggle in submission.csv
def agent(observation, configuration) -> int:
    return AlphaBetaAgentOdd.agent()(observation, configuration)
