# This is a LinkedList implementation of MontyCarlo Tree Search
# but using bitboard_gameovers_heuristic() instead of run_random_simulation()
# Inspired by https://www.kaggle.com/matant/monte-carlo-tree-search-connectx
from struct import Struct

from agents.MontyCarlo.MontyCarloHeuristic import MontyCarloHeuristicNode
from core.ConnectXBBNN import *
from heuristics.BitsquaresHeuristic import bitsquares_heuristic_sigmoid

Hyperparameters = namedtuple('hyperparameters', [])

class MontyCarloOddEvenNode(MontyCarloHeuristicNode):
    persist        = False
    heuristic_fn   = bitsquares_heuristic_sigmoid
    heuristic_args = {}


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
