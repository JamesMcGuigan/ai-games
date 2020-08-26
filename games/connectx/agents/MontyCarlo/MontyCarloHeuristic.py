# This is a LinkedList implementation of MontyCarlo Tree Search
# but using bitboard_gameovers_heuristic() instead of run_random_simulation()
# Inspired by https://www.kaggle.com/matant/monte-carlo-tree-search-connectx
from struct import Struct
from typing import Callable
from typing import Dict

from agents.MontyCarlo.MontyCarloPure import MontyCarloNode
from core.ConnectXBBNN import *
from heuristics.BitboardGameoversHeuristic import bitboard_gameovers_heuristic_sigmoid

Hyperparameters = namedtuple('hyperparameters', [])

class MontyCarloHeuristicNode(MontyCarloNode):
    root_nodes:     List[Union['MontyCarloNode', None]] = [None, None, None]  # root_nodes[observation.mark]
    heuristic_fn:   bitboard_gameovers_heuristic_sigmoid
    heuristic_args: {}

    def __init__(
            self,
            bitboard:      np.ndarray,
            player_id:     int,
            parent:        Union['MontyCarloNode', None] = None,
            parent_action: Union[int,None]              = None,
            exploration:   float = 1.0,
            heuristic_fn:  Callable = None,
            heuristic_args: Dict    = None,
            **kwargs
    ):
        super().__init__(
            bitboard        = bitboard,
            player_id       = player_id,
            parent          = parent,
            parent_action   = parent_action,
            exploration     = exploration,
            heuristic_fn    = heuristic_fn,
            heuristic_args  = heuristic_args,
            **kwargs
        )
        # self.kwargs[] needs to be defined in order to pass arguments down to child nodes
        self.kwargs['heuristic_fn']   = self.heuristic_fn   = heuristic_fn   or self.__class__.heuristic_fn
        self.kwargs['heuristic_args'] = self.heuristic_args = heuristic_args or self.__class__.heuristic_args
        self.heuristic = self.heuristic_fn(**(heuristic_args or {}))

    def simulate(self) -> float:
        score = self.heuristic(self.bitboard, self.player_id)
        return score




def MontyCarloHeuristic(**kwargs):
    # observation   = {'mark': 1, 'board': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    # configuration = {'columns': 7, 'rows': 6, 'inarow': 4, 'steps': 1000, 'timeout': 8}
    def MontyCarloHeuristic(observation: Struct, configuration: Struct) -> int:
        return MontyCarloHeuristicNode.agent(**kwargs)(observation, configuration)
    return MontyCarloHeuristic

def MontyCarloHeuristicKaggle(observation, configuration):
    return MontyCarloHeuristic()(observation, configuration)
