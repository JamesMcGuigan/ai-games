# This is a LinkedList implementation of MontyCarlo Tree Search
# but using bitboard_gameovers_heuristic() instead of run_random_simulation()
# Inspired by https://www.kaggle.com/matant/monte-carlo-tree-search-connectx
from struct import Struct

from agents.MontyCarlo.MontyCarloHeuristic import MontyCarloHeuristicNode
from core.ConnectXBBNN import *
from heuristics.OddEvenHeuristic import oddeven_bitsquares_heuristic_sigmoid

Hyperparameters = namedtuple('hyperparameters', [])

class MontyCarloOddEvenNode(MontyCarloHeuristicNode):
    heuristic_fn   = oddeven_bitsquares_heuristic_sigmoid
    heuristic_args = {}  # reward_power=1.75, reward_3_pair=0, reward_3_endgame=1, reward_2_endgame=0.05, sigmoid_width=6.0, sigmoid_height=1.0


class MontyCarloOddEvenNode2(MontyCarloOddEvenNode):
    pass


def MontyCarloOddEven(**kwargs):
    # observation   = {'mark': 1, 'board': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    # configuration = {'columns': 7, 'rows': 6, 'inarow': 4, 'steps': 1000, 'timeout': 8}
    def MontyCarloBitsquares(observation: Struct, configuration: Struct) -> int:
        return MontyCarloOddEvenNode.agent(**kwargs)(observation, configuration)
    return MontyCarloBitsquares

def MontyCarloOddEvenKaggle(observation, configuration):
    return MontyCarloOddEven()(observation, configuration)
