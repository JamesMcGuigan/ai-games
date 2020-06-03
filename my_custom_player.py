import random
import time
from functools import lru_cache
from itertools import chain
from operator import itemgetter

import numpy as np

from isolation.isolation import Action
from sample_players import BasePlayer



# class CustomPlayer(DataPlayer):
class CustomPlayer(BasePlayer):
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

    search_fn            = 'alphabeta'       # or 'minmax'
    search_max_depth     = 50

    heuristic_fn         = 'heuristic_area'  # or 'heuristic_liberties'
    heuristic_area_depth = 4
    heuristic_area_max   = len(Action) * 3


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.precache()

    # def precache( self, depth=5 ):
    #     state = Isolation()
    #     time_start = time.perf_counter()
    #     action = self.alphabeta(state, depth=depth)  # precache for first move
    #     print( 'precache()', type(action), action, int((time.perf_counter() - time_start) * 1000), 'ms' )

    def get_action(self, state, verbose=False):
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
        if verbose: print()
        for depth in range(1,self.search_max_depth):
            action = self.search(state, depth=depth)
            self.queue.put(action)
            if verbose: print(depth, end=' ')
        if verbose: print( depth, type(action), action, int((time.perf_counter() - time_start) * 1000), 'ms' )


    ### Heuristics

    @classmethod
    @lru_cache(None, typed=True)
    def heuristic(cls, state, player_id):
        if cls.heuristic_fn == 'heuristic_area':
            return cls.heuristic_area(state, player_id)
        if cls.heuristic_fn == 'heuristic_liberties':
            return cls.heuristic_liberties(state, player_id)  # won 45%
        raise NotImplementedError('cls.heuristic_fn must be in ["heuristic_area", "heuristic_liberties"] - got: ', cls.heuristic_fn)

    @classmethod
    @lru_cache(None, typed=True)
    def heuristic_liberties( cls, state, player_id ):
        own_loc = state.locs[player_id]
        opp_loc = state.locs[1 - player_id]
        own_liberties = cls.liberties(state, own_loc)
        opp_liberties = cls.liberties(state, opp_loc)
        return len(own_liberties) - len(opp_liberties)

    @classmethod
    @lru_cache(None, typed=True)
    def heuristic_area( cls, state, player_id):
        own_loc = state.locs[player_id]
        opp_loc = state.locs[1 - player_id]
        own_area = cls.count_area_liberties(state, own_loc)
        opp_area = cls.count_area_liberties(state, opp_loc)
        return own_area - opp_area

    @staticmethod
    @lru_cache(None, typed=True)
    def liberties( state, cell ):
        """add a @lru_cache around this function"""
        return state.liberties(cell)

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
        if self.search_fn == 'minmax':
            return self.minmax(state, depth)
        if self.search_fn == 'alphabeta':
            return self.alphabeta(state, depth)
        raise NotImplementedError('cls.search_fn must be in ["minmax", "alphabeta"] - got: ', self.search_fn)

    ### Search: Minmax

    def minmax( self, state, depth ):
        return max(state.actions(),
                   key=lambda action: self.minmax_min_value(state.result(action), self.player_id, depth-1))

    @classmethod
    @lru_cache(None, typed=True)
    def minmax_min_value(cls, state, player_id, depth):
        if state.terminal_test(): return state.utility(player_id)
        if depth == 0:            return cls.heuristic(state, player_id)
        scores = [
            cls.minmax_max_value(state.result(action), player_id, depth-1)
            for action in state.actions()
        ]
        return min(scores) if len(scores) else np.inf

    @classmethod
    @lru_cache(None, typed=True)
    def minmax_max_value(cls, state, player_id, depth):
        if state.terminal_test(): return state.utility(player_id)
        if depth == 0:            return cls.heuristic(state, player_id)
        scores = [
            cls.minmax_min_value(state.result(action), player_id, depth-1)
            for action in state.actions()
        ]
        return max(scores) if len(scores) else -np.inf


    ### Search: AlphaBeta

    def alphabeta( self, state, depth ):
        actions = state.actions()
        scores  = [
            self.alphabeta_min_value(state.result(action), player_id=self.player_id, depth=depth-1)
            for action in actions
        ]
        score, action = max(zip(scores,actions), key=itemgetter(0))
        return action

    @classmethod
    @lru_cache(None, typed=True)
    def alphabeta_min_value(cls, state, player_id, depth, alpha=-np.inf, beta=np.inf):
        if state.terminal_test(): return state.utility(player_id)
        if depth == 0:            return cls.heuristic(state, player_id)
        score = np.inf
        for action in state.actions():
            result     = state.result(action)
            score      = min(score, cls.alphabeta_max_value(result, player_id, depth-1, alpha, beta))
            if score <= alpha: return score
            beta       = min(beta,score)
        return score

    @classmethod
    @lru_cache(None, typed=True)
    def alphabeta_max_value(cls, state, player_id, depth, alpha=-np.inf, beta=np.inf):
        if state.terminal_test(): return state.utility(player_id)
        if depth == 0:            return cls.heuristic(state, player_id)
        score = -np.inf
        for action in state.actions():
            result     = state.result(action)
            score      = max(score, cls.alphabeta_min_value(result, player_id, depth-1, alpha, beta))
            if score >= beta: return score
            alpha      = max(alpha, score)
        return score
