import math

from isolation import DebugState
from sample_players import BasePlayer



class DistancePlayer(BasePlayer):
    """ Player that chooses next move to maximize heuristic score. This is
    equivalent to a minimax search agent with a search depth of one.
    """
    def distance(self, state):
        if None in state.locs: return 0
        own_xy  = DebugState.ind2xy(state.locs[self.player_id])
        opp_xy  = DebugState.ind2xy(state.locs[1-self.player_id])
        manhattan_distance = abs(own_xy[0]-opp_xy[0]) + abs(own_xy[1]-opp_xy[1])
        euclidean_distance = math.sqrt( (own_xy[0]-opp_xy[0])**2 + (own_xy[1]-opp_xy[1])**2 )
        return euclidean_distance

    def score(self, state):
        return self.distance(state)

    def get_action(self, state):
        self.queue.put(max(state.actions(), key=lambda x: self.score(state.result(x))))



class GreedyDistancePlayer(DistancePlayer):
    """ Player that chooses next move to maximize heuristic score. This is
    equivalent to a minimax search agent with a search depth of one.
    """
    def liberty_difference(self, state):
        own_loc       = state.locs[self.player_id]
        opp_loc       = state.locs[self.player_id]
        own_liberties = len(state.liberties(own_loc))
        opp_liberties = len(state.liberties(own_loc))
        score = own_liberties - opp_liberties
        return score

    def score(self, state):
        distance  = self.distance(state)
        liberties = self.liberty_difference(state)
        score     = distance + liberties
        # print('{:.2f} {:.2f} {:.2f}'.format(score, distance, liberties))
        return score

    def get_action(self, state):
        self.queue.put(max(state.actions(), key=lambda x: self.score(state.result(x))))
