# This is a LinkedList implementation of MontyCarlo Tree Search
# but using bitboard_gameovers_heuristic() instead of run_random_simulation()
# Inspired by https://www.kaggle.com/matant/monte-carlo-tree-search-connectx
from struct import Struct

from agents.MontyCarlo.MontyCarloPure import MontyCarloNode
from core.ConnectXBBNN import *
from heuristics.BitboardGameoversHeuristic import bitboard_gameovers_heuristic
from util.sigmoid import scaled_sigmoid

Hyperparameters = namedtuple('hyperparameters', [])

class MontyCarloHeuristicNode(MontyCarloNode):
    root_nodes: List[Union['MontyCarloNode', None]] = [None, None, None]  # root_nodes[observation.mark]

    def __init__(
            self,
            bitboard:      np.ndarray,
            player_id:     int,
            parent:        Union['MontyCarloNode', None] = None,
            parent_action: Union[int,None]              = None,
            exploration:     float = 1.0,
            heuristic_scale: float = 6.0,  # 6 seems to score best against other values
            **kwargs
    ):
        super().__init__(
            bitboard        = bitboard,
            player_id       = player_id,
            parent          = parent,
            parent_action   = parent_action,
            exploration     = exploration,
            heuristic_scale = heuristic_scale,  # saved as self.kwargs in parent constructor
            **kwargs
        )
        self.heuristic_scale = heuristic_scale


    def simulate(self) -> float:
        score = bitboard_gameovers_heuristic(self.bitboard, self.player_id)
        score = scaled_sigmoid(score, self.heuristic_scale)
        return score




def MontyCarloHeuristic(**kwargs):
    # observation   = {'mark': 1, 'board': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    # configuration = {'columns': 7, 'rows': 6, 'inarow': 4, 'steps': 1000, 'timeout': 8}
    def MontyCarloHeuristic(observation: Struct, configuration: Struct) -> int:
        return MontyCarloHeuristicNode.agent(observation, configuration, **kwargs)
    return MontyCarloHeuristic

def MontyCarloHeuristicKaggle(observation, configuration):
    return MontyCarloHeuristic()(observation, configuration)
