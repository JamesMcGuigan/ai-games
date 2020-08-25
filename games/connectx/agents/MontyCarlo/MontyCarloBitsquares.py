# This is a LinkedList implementation of MontyCarlo Tree Search
# but using bitboard_gameovers_heuristic() instead of run_random_simulation()
# Inspired by https://www.kaggle.com/matant/monte-carlo-tree-search-connectx
from struct import Struct

from agents.MontyCarlo.MontyCarloPure import MontyCarloNode
from core.ConnectXBBNN import *
from heuristics.BitsquaresHeuristic import bitsquares_heuristic_sigmoid

Hyperparameters = namedtuple('hyperparameters', [])

class MontyCarloBitsquaresNode(MontyCarloNode):
    persist   = False
    root_nodes: List[Union['MontyCarloNode', None]] = [None, None, None]  # root_nodes[observation.mark]

    def __init__(
            self,
            bitboard:      np.ndarray,
            player_id:     int,
            parent:        Union['MontyCarloNode', None] = None,
            parent_action: Union[int,None]              = None,
            exploration:   float = 1.0,
            **kwargs
    ):
        super().__init__(
            bitboard        = bitboard,
            player_id       = player_id,
            parent          = parent,
            parent_action   = parent_action,
            exploration     = exploration,
            **kwargs
        )
        self.heuristic = bitsquares_heuristic_sigmoid()


    def simulate(self) -> float:
        score = self.heuristic(self.bitboard, self.player_id)
        return score

class MontyCarloBitsquaresNode2(MontyCarloBitsquaresNode):
    pass


def MontyCarloBitsquares(**kwargs):
    # observation   = {'mark': 1, 'board': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    # configuration = {'columns': 7, 'rows': 6, 'inarow': 4, 'steps': 1000, 'timeout': 8}
    def MontyCarloBitsquares(observation: Struct, configuration: Struct) -> int:
        return MontyCarloBitsquaresNode.agent(**kwargs)(observation, configuration)
    return MontyCarloBitsquares

def MontyCarloBitsquaresKaggle(observation, configuration):
    return MontyCarloBitsquares()(observation, configuration)
