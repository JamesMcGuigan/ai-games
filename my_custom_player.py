import random
import time
from operator import itemgetter

from sample_players import BasePlayer, DataPlayer
import numpy as np

class CustomPlayer(DataPlayer):
# class CustomPlayer(BasePlayer):
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
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
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

        # Iterative deepening
        time_start = time.perf_counter()
        for depth in range(1,100):
            # action = self.minmax(state,         depth=depth)
            action, score = self.alphabeta(state, depth=depth)  # very slow to run for first move
            self.queue.put(action)
        if verbose:
            print( type(action), action, int((time.perf_counter() - time_start) * 1000), 'ms' )


    def heuristic(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - len(opp_liberties)


    def random_agent( self, state ):
        return random.choice(state.actions())

    def minmax( self, state, depth=np.inf ):
        def min_value(state, depth):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth == 0:            return self.heuristic(state)
            scores = [
                max_value(state.result(action), depth-1)
                for action in state.actions()
            ]
            return min(scores) if len(scores) else np.inf

        def max_value(state, depth):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth == 0:            return self.heuristic(state)
            scores = [
                max_value(state.result(action), depth-1)
                for action in state.actions()
            ]
            return max(scores) if len(scores) else -np.inf

        # return max(state.actions(), key=lambda action: min_value(state.result(action), depth-1))
        actions = state.actions()
        scores  = [ min_value(state.result(action), depth=depth-1) for action in actions ]
        score, action = max(zip(scores,actions), key=itemgetter(0))
        return action, score


    def alphabeta( self, state, depth ):
        def min_value(state, depth, alpha=-np.inf, beta=np.inf):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth == 0:            return self.heuristic(state)
            score = np.inf
            for action in state.actions():
                result     = state.result(action)
                score      = min(score, max_value(result, depth-1, alpha, beta))
                if score <= alpha: return score
                beta       = min(beta,score)
            return score

        def max_value(state, depth, alpha=-np.inf, beta=np.inf):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth == 0:            return self.heuristic(state)
            score = -np.inf
            for action in state.actions():
                result     = state.result(action)
                score      = max(score, max_value(result, depth-1, alpha, beta))
                if score >= beta: return score
                alpha      = max(alpha, score)
            return score

        actions = state.actions()
        scores  = [ min_value(state.result(action), depth=depth-1) for action in actions ]
        score, action = max(zip(scores,actions), key=itemgetter(0))
        return action, score
