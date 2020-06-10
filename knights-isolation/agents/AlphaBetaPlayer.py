import math
import random
import time
from functools import lru_cache
from itertools import chain
from operator import itemgetter

from agents.DataSavePlayer import DataSavePlayer



class AlphaBetaPlayer(DataSavePlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the cls.context attribute.
    **********************************************************************
    """

    verbose_depth        = False
    search_fn            = 'alphabeta'       # or 'minimax'
    search_max_depth     = 50

    heuristic_fn         = 'heuristic_liberties'  # 'heuristic_liberties' | 'heuristic_area'
    heuristic_area_depth = 4                      # 4 seems to be the best number against LibertiesPlayer
    heuristic_area_max   = 8 * 5  # 8=len(Action) # 5 seems to be the best number against LibertiesPlayer

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.precache()

    # def precache( self, depth=5 ):
    #     state = Isolation()
    #     time_start = time.perf_counter()
    #     action = self.alphabeta(state, depth=depth)  # precache for first move
    #     print( 'precache()', type(action), action, int((time.perf_counter() - time_start) * 1000), 'ms' )

    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE:
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)

        time_start = time.perf_counter()
        action = random.choice(state.actions())
        self.queue.put( action )     # backup move incase of early timeout

        # The real trick with iterative deepening is caching, which allows us to out-depth the default minimax Agent
        if self.verbose_depth: print('\n'+ self.__class__.__name__.ljust(20) +' | depth:', end=' ', flush=True)
        for depth in range(1, self.search_max_depth+1):
            action, score = self.search(state, depth=depth)
            self.queue.put(action)
            if self.verbose_depth: print(depth, end=' ', flush=True)
            if abs(score) == math.inf:
                if self.verbose_depth: print(score, end=' ', flush=True)
                break  # terminate iterative deepening on inescapable victory condition
        # if self.verbose_depth: print( depth, type(action), action, int((time.perf_counter() - time_start) * 1000), 'ms' )


    ### Heuristics

    def heuristic(self, state, player_id):
        if self.heuristic_fn == 'heuristic_area':
            return self.heuristic_area(state, player_id)
        if self.heuristic_fn == 'heuristic_liberties':
            return self.heuristic_liberties(state, player_id)  # won 45%
        raise NotImplementedError('cls.heuristic_fn must be in ["heuristic_area", "heuristic_liberties"] - got: ', self.heuristic_fn)


    # Its not worth persisting this cache to disk, when too large (million+) its becomes quicker to compute than lookup
    @classmethod
    @lru_cache(None, typed=True)
    def heuristic_liberties(cls, state, player_id):
        own_loc = state.locs[player_id]
        opp_loc = state.locs[1 - player_id]
        own_liberties = cls.liberties(state, own_loc)
        opp_liberties = cls.liberties(state, opp_loc)
        return len(own_liberties) - len(opp_liberties)

    @staticmethod
    @lru_cache(None, typed=True)
    def liberties( state, cell ):
        """add a @lru_cache around this function"""
        return state.liberties(cell)


    @classmethod
    def heuristic_area(cls, state, player_id):
        """ Persistent caching to disk """
        return cls.cache(cls._heuristic_area, state, player_id)
    @classmethod
    def _heuristic_area(self, state, player_id):
        own_loc  = state.locs[player_id]
        opp_loc  = state.locs[1 - player_id]
        own_area = self.count_area_liberties(state, own_loc)
        opp_area = self.count_area_liberties(state, opp_loc)
        return own_area - opp_area

    @classmethod
    @lru_cache(None, typed=True)  # depth > 1 exceeds 150ms timeout (without caching)
    def count_area_liberties( cls, state, start_loc ):
        depth     = cls.heuristic_area_depth
        max_area  = cls.heuristic_area_max

        area      = set()
        frontier  = { start_loc }
        seen      = set()
        while len(frontier) and len(area) < max_area and depth > 0:
            seen     |= frontier
            frontier |= set(chain(*[ cls.liberties(state, cell) for cell in frontier ]))
            area     |= frontier
            frontier -= seen
            depth    -= 1
        return len(area)



    ### Search

    def search( self, state, depth ):
        if self.search_fn == 'minimax':
            return self.minimax(state, depth)
        if self.search_fn == 'alphabeta':
            return self.alphabeta(state, depth)
        raise NotImplementedError('cls.search_fn must be in ["minimax", "alphabeta"] - got: ', self.search_fn)


    ### Search: Minmax

    def minimax( self, state, depth ):
        actions = state.actions()
        scores  = [
            self.minimax_min_value(state.result(action), player_id=self.player_id, depth=depth-1)
            for action in actions
        ]
        action, score = max(zip(actions, scores), key=itemgetter(1))
        return action, score

    def minimax_min_value(self, state, player_id, depth):
        return self.cache_infinite(self._minimax_min_value, state, player_id, depth)
    def _minimax_min_value( self, state, player_id, depth ):
        if state.terminal_test(): return state.utility(player_id)
        if depth == 0:            return self.heuristic(state, player_id)
        scores = [
            self.minimax_max_value(state.result(action), player_id, depth - 1)
            for action in state.actions()
        ]
        return min(scores) if len(scores) else math.inf

    def minimax_max_value(self, state, player_id, depth):
        return self.cache_infinite(self._minimax_max_value, state, player_id, depth)
    def _minimax_max_value( self, state, player_id, depth ):
        if state.terminal_test(): return state.utility(player_id)
        if depth == 0:            return self.heuristic(state, player_id)
        scores = [
            self.minimax_min_value(state.result(action), player_id, depth - 1)
            for action in state.actions()
        ]
        return max(scores) if len(scores) else -math.inf


    ### Search: AlphaBeta

    def alphabeta( self, state, depth ):
        actions = state.actions()
        scores  = [
            self.alphabeta_min_value(state.result(action), player_id=self.player_id, depth=depth-1)
            for action in actions
        ]
        action, score = max(zip(actions, scores), key=itemgetter(1))
        return action, score

    def alphabeta_min_value(self, state, player_id, depth, alpha=-math.inf, beta=math.inf):
        return self.cache_infinite(self._alphabeta_min_value, state, player_id, depth, alpha, beta)
    def _alphabeta_min_value(self, state, player_id, depth, alpha=-math.inf, beta=math.inf):
        if state.terminal_test(): return state.utility(player_id)
        if depth == 0:            return self.heuristic(state, player_id)
        score = math.inf
        for action in state.actions():
            result    = state.result(action)
            score     = min(score, self.alphabeta_max_value(result, player_id, depth-1, alpha, beta))
            if score <= alpha: return score
            beta      = min(beta,score)
        return score

    def alphabeta_max_value(self, state, player_id, depth, alpha=-math.inf, beta=math.inf):
        return self.cache_infinite(self._alphabeta_max_value, state, player_id, depth, alpha, beta)
    def _alphabeta_max_value(self, state, player_id, depth, alpha=-math.inf, beta=math.inf):
        if state.terminal_test(): return state.utility(player_id)
        if depth == 0:            return self.heuristic(state, player_id)
        score = -math.inf
        for action in state.actions():
            result    = state.result(action)
            score     = max(score, self.alphabeta_min_value(result, player_id, depth-1, alpha, beta))
            if score >= beta: return score
            alpha     = max(alpha, score)
        return score



class AlphaBetaAreaPlayer(AlphaBetaPlayer):
    heuristic_fn = 'heuristic_area'  # or 'heuristic_liberties'


# This matches the specs of the course implementation - but this version is better performance optimized
class MinimaxPlayer(AlphaBetaPlayer):
    verbose          = False
    search_fn        = 'minimax'              # or 'alphabeta'
    heuristic_fn     = 'heuristic_liberties'  # 'heuristic_liberties' | 'heuristic_area'
    search_max_depth = 3



# CustomPlayer is the agent exported to the submission
class CustomPlayer(AlphaBetaAreaPlayer):
    pass
