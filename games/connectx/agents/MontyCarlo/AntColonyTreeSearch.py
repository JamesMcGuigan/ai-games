# AntColonyTreeSearch differs from MontyCarloTreeSearch in the choice of the node selection algorithm
# AntColonyTreeSearch used weighted randomness of accumulated scores, rather than UCB1 (Upper Confidence Bound)
import random
from struct import Struct

from agents.MontyCarlo.MontyCarloPure import MontyCarloNode
from core.ConnectXBBNN import *
from heuristics.BitboardGameoversHeuristic import bitboard_gameovers_heuristic
from util.sigmoid import scaled_sigmoid

Hyperparameters = namedtuple('hyperparameters', [])

class AntColonyTreeSearchNode(MontyCarloNode):
    def __init__(
            self,
            bitboard:      np.ndarray,
            player_id:     int,
            parent:        Union['MontyCarloNode', None] = None,
            parent_action: Union[int,None]       = None,
            heuristic_scale:  float = 6.0,   # 6 seems to score best against other values
            start_pheromones: float = 1.0,
            pheromone_power:  float = 1.25,
            **kwargs
    ):
        self.pheromone_power  = pheromone_power
        self.start_pheromones = start_pheromones

        super().__init__(
            bitboard         = bitboard,
            player_id        = player_id,
            parent           = parent,
            parent_action    = parent_action,
            heuristic_scale  = heuristic_scale,
            start_pheromones = start_pheromones,
            pheromone_power  = pheromone_power,  # saved in self.kwargs in parent constructor
            **kwargs
        )
        self.heuristic_scale  = heuristic_scale
        self.total_score      = 0.0
        self.pheromone_score  = start_pheromones
        self.pheromones_total = 0.0


    # BUGFIX: this makes .prune() work without total_count
    @classmethod
    def prune(cls, node: 'MontyCarloNode', min_visits=7, pruned_count=0, total_count=0):
        for n, child in enumerate(node.children):
            if child is None: continue
            if child.total_score < min_visits:
                pruned_count    += child.total_score  # excepting terminal states, this equals the number of grandchildren
                total_count     += child.total_score  # excepting terminal states, this equals the number of grandchildren
                node.children[n] = None
                node.is_expanded = False  # Use def expand(self) to reinitalize state
            else:
                total_count += 1
                pruned_count, total_count = cls.prune(child, min_visits, pruned_count, total_count)
        return pruned_count, total_count


    def simulate(self) -> float:
        score = bitboard_gameovers_heuristic(self.bitboard, self.player_id)
        score = scaled_sigmoid(score, self.heuristic_scale)
        return score


    def backpropagate(self, score: float):
        # child.simulate()  returns score for the player 2
        # child.total_score is accessed via the parent node, so score on this node is from the perspective of player 1
        node = self
        while node is not None:
            score                = self.opponents_score(score)
            node.total_score    += score

            previous_pheromones  = node.pheromone_score
            node.pheromone_score = self.start_pheromones + node.total_score ** self.pheromone_power

            if node.parent is not None and node.parent.is_expanded:
                # Profiler reports that addition [8% runtime] rather than sum() [18% runtime] results in a 2.5x speedup,
                node.parent.pheromones_total += (node.pheromone_score - previous_pheromones)
                # assert np.math.isclose(node.parent.pheromones_total, sum([ node.parent.children[action].pheromone_score for action in node.parent.legal_moves ]), rel_tol=1e-6 ), f"{node.parent.pheromones_total} != {sum([ node.parent.children[action].pheromone_score for action in node.parent.legal_moves ])}"
            node = node.parent  # when we reach the root: node.parent == None which terminates

    def on_expanded(self) -> None:
        super().on_expanded()
        self.pheromones_total = sum([ self.children[action].pheromone_score for action in self.legal_moves ])

    def get_exploration_action(self) -> int:
        # Random choice weighted by pheromone_score
        rand = random.random() * self.pheromones_total
        for action in self.legal_moves:
            assert self.children[action] is not None
            weight = self.children[action].pheromone_score
            if rand > weight:
                rand -= weight
            else:
                return action
        return self.legal_moves[-1]  # This should never happen


def AntColonyTreeSearch(**kwargs):
    # observation   = {'mark': 1, 'board': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    # configuration = {'columns': 7, 'rows': 6, 'inarow': 4, 'steps': 1000, 'timeout': 8}
    def AntColonyTreeSearch(observation: Struct, configuration: Struct) -> int:
        return AntColonyTreeSearchNode.agent(**kwargs)(observation, configuration)
    return AntColonyTreeSearch

def AntColonyTreeSearchKaggle(observation, configuration):
    return AntColonyTreeSearch()(observation, configuration)
