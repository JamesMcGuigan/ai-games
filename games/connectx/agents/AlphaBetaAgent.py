from __future__ import annotations

import math
import random
import time
from operator import itemgetter
from queue import LifoQueue

from games.connectx.core.ConnectX import ConnectX
from games.connectx.core.KaggleGame import KaggleGame
from games.connectx.core.PersistentCacheAgent import PersistentCacheAgent
from util.call_with_timeout_ms import call_with_timeout



class AlphaBetaAgent(PersistentCacheAgent):
    defaults = {
        "verbose_depth":    True,
        "search_max_depth": 5,
    }

    def __init__( self, game: ConnectX, *args, **kwargs ):
        super().__init__(*args, **kwargs)
        self.kwargs    = { **self.defaults, **kwargs }
        self.game      = game
        self.player_id = game.observation.mark
        self.queue     = LifoQueue()
        self.verbose_depth    = self.kwargs.get('verbose_depth')
        self.search_max_depth = self.kwargs.get('search_max_depth')


    ### Exported Interface

    # observation   = {'mark': 1, 'board': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    # configuration = {'columns': 7, 'rows': 6, 'inarow': 4, 'steps': 1000, 'timeout': 2}
    @staticmethod
    def agent(observation, configuration) -> int:
        time_start = time.perf_counter()
        timeout    = configuration.timeout * 0.9  # 200ms spare to return answers

        game    = ConnectX(observation, configuration)
        agent   = AlphaBetaAgent(game)
        action  = agent.get_action(timeout - (time.perf_counter()-time_start))
        return action



    ### Public Interface

    def get_action( self, timeout: float ):
        print(f'get_action({timeout})')
        call_with_timeout(timeout, self.iterative_deepening_search)
        if self.queue.empty():
            action = random.choice(self.game.actions)  # backup move incase of early timeout
        else:
            action = self.queue.get()
        return action



    ### Search Functions

    def iterative_deepening_search( self ):
        # The real trick with iterative deepening is caching, which allows us to out-depth the default minimax Agent
        if self.verbose_depth: print('\n'+ self.__class__.__name__.ljust(20) +' | depth:', end=' ', flush=True)
        for depth in range(1, self.search_max_depth+1):
            action, score = self.alphabeta(self.game, depth=depth)
            self.queue.put(action)
            if self.verbose_depth: print(depth, end=' ', flush=True)
            if abs(score) == math.inf:
                if self.verbose_depth: print(score, end=' ', flush=True)
                break  # terminate iterative deepening on inescapable victory condition
        # if self.verbose_depth: print( depth, type(action), action, int((time.perf_counter() - time_start) * 1000), 'ms' )


    def alphabeta( self, game, depth ):
        actions = game.actions
        scores  = [
            self.alphabeta_min_value(game.result(action), player_id=self.player_id, depth=depth - 1)
            for action in actions
        ]
        action, score = max(zip(actions, scores), key=itemgetter(1))
        return action, score

    def alphabeta_min_value( self, game: KaggleGame, player_id: int, depth: int, alpha=-math.inf, beta=math.inf ):
        return self.cache_infinite(self._alphabeta_min_value, game, player_id, depth, alpha, beta)

    def _alphabeta_min_value( self, game: KaggleGame, player_id, depth: int, alpha=-math.inf, beta=math.inf ):
        if game.gameover: return game.utility(player_id)
        if depth == 0:    return game.score(player_id)
        score = math.inf
        for action in game.actions:
            result    = game.result(action)
            score     = min(score, self.alphabeta_max_value(result, player_id, depth-1, alpha, beta))
            if score <= alpha: return score
            beta      = min(beta,score)
        return score

    def alphabeta_max_value( self, game: KaggleGame, player_id: int, depth, alpha=-math.inf, beta=math.inf ):
        return self.cache_infinite(self._alphabeta_max_value, game, player_id, depth, alpha, beta)

    def _alphabeta_max_value( self, game: KaggleGame, player_id: int, depth, alpha=-math.inf, beta=math.inf ):
        if game.gameover:  return game.utility(player_id)
        if depth == 0:     return game.score(player_id)
        score = -math.inf
        for action in game.actions:
            result    = game.result(action)
            score     = max(score, self.alphabeta_min_value(result, player_id, depth-1, alpha, beta))
            if score >= beta: return score
            alpha     = max(alpha, score)
        return score



# The last function defined in the file run by Kaggle in submission.csv
def agent(observation, configuration) -> int:
    return AlphaBetaAgent.agent(observation, configuration)
