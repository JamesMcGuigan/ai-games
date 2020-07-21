import random
import time
from queue import LifoQueue

from core.ConnectX import ConnectX
from util.call_with_timeout_ms import call_with_timeout



class RandomAgent:
    def __init__( self, game: ConnectX, *args, **kwargs ):
        super().__init__()
        self.game      = game
        self.player_id = game.observation.mark
        self.queue     = LifoQueue()


    ### Exported Interface

    # observation   = {'mark': 1, 'board': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    # configuration = {'columns': 7, 'rows': 6, 'inarow': 4, 'steps': 1000, 'timeout': 2}
    @staticmethod
    def agent(observation, configuration, **kwargs) -> int:
        time_start = time.perf_counter()
        timeout    = configuration.timeout * 0.8  # 400ms spare to return answers - 0.9 is too small

        game    = ConnectX(observation, configuration)
        agent   = RandomAgent(game, **kwargs)
        action  = agent.get_action(timeout - (time.perf_counter()-time_start))
        return int(action)


    ### Public Interface

    def get_action( self, timeout: float ) -> int:
        call_with_timeout(timeout, self.search)
        action = self.queue.get()
        return int(action)

    def search( self ):
        action = random.choice(self.game.actions)  # backup move incase of early timeout
        self.queue.put(action)



# The last function defined in the file run by Kaggle in submission.csv
def agent(observation, configuration) -> int:
    return RandomAgent.agent(observation, configuration)
