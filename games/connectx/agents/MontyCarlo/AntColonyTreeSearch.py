# AntColonyTreeSearch differs from MontyCarloTreeSearch in the choice of the node selection algorithm
# AntColonyTreeSearch used weighted randomness of accumulated scores, rather than UCB1 (Upper Confidence Bound)
import random
from struct import Struct

from agents.MontyCarlo.MontyCarloHeuristic import MontyCarloHeuristicNode
from agents.MontyCarlo.MontyCarloLinkedList import MontyCarloNode
from core.ConnectXBBNN import *

Hyperparameters = namedtuple('hyperparameters', [])

class AntColonyTreeSearchNode(MontyCarloHeuristicNode):
    root_nodes: List[Union['MontyCarloNode', None]] = [None, None, None]  # root_nodes[observation.mark]

    def __init__(
            self,
            bitboard:      np.ndarray,
            player_id:     int,
            configuration: Configuration,
            parent:        Union['MontyCarloNode', None] = None,
            parent_action: Union[int,None]       = None,
            start_pheromones: float = 1.0,
            pheromone_power:  float = 1.25,
            **kwargs
    ):
        self.pheromone_power  = pheromone_power
        self.start_pheromones = start_pheromones

        super().__init__(
            bitboard         = bitboard,
            player_id        = player_id,
            configuration    = configuration,
            parent           = parent,
            parent_action    = parent_action,
            start_pheromones = start_pheromones,
            pheromone_power  = pheromone_power,  # saved in self.kwargs in parent constructor
            **kwargs
        )

        self.total_score      = 0.0
        self.pheromone_score  = start_pheromones
        self.pheromones_total = 0.0


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
                if node.parent.pheromones_total == 0.0:
                    node.parent.pheromones_total = sum([ node.parent.children[action].pheromone_score for action in node.parent.legal_moves ])
                else:
                    node.parent.pheromones_total += (node.pheromone_score - previous_pheromones)
                # assert np.math.isclose(node.parent.pheromones_total, sum([ node.parent.children[action].pheromone_score for action in node.parent.legal_moves ]), rel_tol=1e-6 ), f"{node.parent.pheromones_total} != {sum([ node.parent.children[action].pheromone_score for action in node.parent.legal_moves ])}"
            node = node.parent  # when we reach the root: node.parent == None which terminates


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
        return AntColonyTreeSearchNode.agent(observation, configuration, **kwargs)
    return AntColonyTreeSearch

def AntColonyTreeSearchKaggle(observation, configuration):
    return AntColonyTreeSearch()(observation, configuration)
