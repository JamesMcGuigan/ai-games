import math
import random
import time
from queue import LifoQueue
from struct import Struct
from typing import Callable

from games.connectx.core.ConnectX import ConnectX
from games.connectx.core.KaggleGame import KaggleGame
from games.connectx.core.PersistentCacheAgent import PersistentCacheAgent
from games.connectx.heuristics.LinesHeuristic import LinesHeuristic



class AlphaBetaAgent(PersistentCacheAgent):
    heuristic_class = LinesHeuristic
    defaults = {
        "verbose_depth":    True,
        "search_max_depth": 100 # if "pytest" not in sys.modules else 3,
    }

    def __init__( self, game: ConnectX, *args, **kwargs ):
        super().__init__(*args, **kwargs)
        self.kwargs    = { **self.defaults, **kwargs }
        self.game      = game
        self.player_id = game.observation.mark
        self.queue     = LifoQueue()
        self.verbose_depth    = self.kwargs.get('verbose_depth')
        self.search_max_depth = self.kwargs.get('search_max_depth')


    ### Public Interface

    def get_action( self, endtime: float ) -> int:
        action = self.iterative_deepening_search(endtime=endtime)
        return int(action)



    ### Search Functions

    def iterative_deepening_search( self, endtime=0.0 ) -> int:
        # The real trick with iterative deepening is caching, which allows us to out-depth the default minimax Agent
        if self.verbose_depth: print('\n'+ self.__class__.__name__.ljust(20) +' | depth:', end=' ', flush=True)
        best_action = random.choice(self.game.actions)
        for depth in range(1, self.search_max_depth+1):
            action, score = self.alphabeta(self.game, depth=depth, endtime=endtime)
            if endtime and time.perf_counter() >= endtime: break  # ignore results on timeout

            best_action = action
            if self.verbose_depth: print(depth, end=' ', flush=True)
            if abs(score) == math.inf:
                if self.verbose_depth: print(score, end=' ', flush=True)
                break  # terminate iterative deepening on inescapable victory condition

        return best_action
        # if self.verbose_depth: print( depth, type(action), action, int((time.perf_counter() - time_start) * 1000), 'ms' )


    def alphabeta( self, game, depth, endtime=0.0 ):
        scores = []
        best_action = random.choice(game.actions)
        best_score  = -math.inf
        for action in game.actions:
            result = game.result(action)
            score  = self.alphabeta_min_value(result, player_id=self.player_id, depth=depth-1, endtime=endtime)
            if endtime and time.perf_counter() >= endtime: break
            if score > best_score:
                best_score  = score
                best_action = action
            scores.append(score)  # for debugging

        # action, score = max(zip(game.actions, scores), key=itemgetter(1))
        return best_action, best_score  # This is slightly quicker for timeout purposes


    def alphabeta_min_value( self, game: KaggleGame, player_id: int, depth: int, alpha=-math.inf, beta=math.inf, endtime=0.0):
        return self.cache_infinite(self._alphabeta_min_value, game, player_id, depth, alpha, beta, endtime)
    def _alphabeta_min_value( self, game: KaggleGame, player_id, depth: int, alpha=-math.inf, beta=math.inf, endtime=0.0 ):
        sign = 1 if player_id != game.player_id else -1
        if game.gameover:  return sign * game.heuristic.utility  # score relative to previous player who made the move
        if depth == 0:     return sign * game.heuristic.score
        scores = []
        score  = math.inf
        for action in game.actions:
            result    = game.result(action)
            score     = min(score, self.alphabeta_max_value(result, player_id, depth-1, alpha, beta, endtime))
            if endtime and time.perf_counter() >= endtime: return score
            if score <= alpha: return score
            beta      = min(beta,score)
            scores.append(score)  # for debugging
        return score

    def alphabeta_max_value( self, game: KaggleGame, player_id: int, depth, alpha=-math.inf, beta=math.inf, endtime=0.0  ):
        return self.cache_infinite(self._alphabeta_max_value, game, player_id, depth, alpha, beta, endtime)
    def _alphabeta_max_value( self, game: KaggleGame, player_id: int, depth, alpha=-math.inf, beta=math.inf, endtime=0.0  ):
        sign = 1 if player_id != game.player_id else -1
        if game.gameover:  return sign * game.heuristic.utility  # score relative to previous player who made the move
        if depth == 0:     return sign * game.heuristic.score
        scores = []
        score  = -math.inf
        for action in game.actions:
            result    = game.result(action)
            score     = max(score, self.alphabeta_min_value(result, player_id, depth-1, alpha, beta, endtime))
            if endtime and time.perf_counter() >= endtime: return score
            if score >= beta: return score
            alpha     = max(alpha, score)
            scores.append(score)  # for debugging
        return score



    ### Exported Interface

    # observation   = {'mark': 1, 'board': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    # configuration = {'columns': 7, 'rows': 6, 'inarow': 4, 'steps': 1000, 'timeout': 2}
    @classmethod
    def agent(cls, **kwargs) -> Callable[[Struct, Struct],int]:
        heuristic_class = kwargs.get('heuristic_class', cls.heuristic_class)

        def kaggle_agent(observation: Struct, configuration: Struct):
            endtime = time.perf_counter() + configuration.timeout - 1.1  # Leave a small amount of time to return an answer
            game    = ConnectX(observation, configuration, heuristic_class, **kwargs)
            agent   = cls(game, **kwargs)
            action  = agent.get_action(endtime)
            # print(endtime - time.perf_counter(), 's')  # min -0.001315439000000751 s
            return int(action)
        return kaggle_agent



# The last function defined in the file run by Kaggle in submission.csv
def agent(observation, configuration) -> int:
    return AlphaBetaAgent.agent()(observation, configuration)
