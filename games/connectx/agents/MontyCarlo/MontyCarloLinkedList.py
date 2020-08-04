# This is a LinkedList implementation of MontyCarlo Tree Search
# Inspired by https://www.kaggle.com/matant/monte-carlo-tree-search-connectx
import time
from struct import Struct
from typing import List
from typing import Union

from core.ConnectXBBNN import *



Hyperparameters = namedtuple('hyperparameters', [])

class MCNode:
    def __init__(
            self,
            bitboard:      np.ndarray,
            player_id:     int,
            config:        Configuration,
            parent:        Union['MCNode', None] = None,
            parent_action: Union[int,None]       = None,
            exploration: float = 1.0,
            # action_policy: str = 'max'  # 'max' or 'random'
    ):
        self.bitboard      = bitboard
        self.player_id     = player_id
        self.next_player   = 3 - player_id

        self.config        = config
        self.exploration   = exploration

        self.mirror_hash   = hash_bitboard(bitboard)
        self.legal_moves   = get_legal_moves(bitboard)
        self.is_gameover   = is_gameover(bitboard)
        self.winner        = get_winner(bitboard) if self.is_gameover else 0
        self.utility       = 1 if self.winner == self.player_id else 0  # Scores in range 0-1

        self.parent        = parent
        self.parent_action = parent_action
        self.children: List[Union[MCNode,None]] = [ None for action in range(config.columns) ]  # include illegal moves to preserve indexing
        self.total_score   = 0.0
        self.total_visits  = 0
        self.is_expanded   = False


    ### Constructors and Lookups

    def create_child( self, action: int ) -> 'MCNode':
        result = result_action(self.bitboard, action, self.player_id)
        child  = MCNode(
            bitboard      = result,
            player_id     = next_player_id(self.player_id),
            parent        = self,
            parent_action = action,
            config        = self.config,
            exploration   = self.exploration
        )
        self.children[action] = child
        self.is_expanded      = self._is_expanded()
        return child


    def find_child( self, bitboard: np.array ) -> Union['MCNode', None]:
        if np.all( self.bitboard == bitboard ):
            return self
        for child in self.children:
            if child is None: continue
            if np.all( child.bitboard == bitboard ):
                return child
        return None



    ### Properties

    def _is_expanded(self) -> bool:
        is_expanded = True
        for action in self.legal_moves:
            if self.children[action] is None:
                is_expanded = False
                break
        return is_expanded


    def get_unexpanded(self) -> List[int]:
        return [
            action
            for action in self.legal_moves
            if self.children[action] is None
        ]


    ### Action Selection

    def get_best_action(self) -> int:
        scores = [ self.children[action].total_score
                   if self.children[action] is not None else 0
                   for action in self.legal_moves ]
        index  = np.argmax(scores)
        action = self.legal_moves[index]
        return action


    def get_exploration_action(self) -> int:
        scores = [ self.ucb1_score( self.children[action] )
                   for action in self.legal_moves ]
        index  = np.argmax(scores)
        action = self.legal_moves[index]
        return action



    ### Scores

    @staticmethod
    def ucb1_score(node) -> float:
        if node is None or node.total_visits == 0:
            return np.inf
        else:
            score = node.total_score / node.total_visits
            if node.parent is not None:
                score += (
                    node.exploration * np.sqrt(2)
                    * np.log(node.parent.total_visits) / node.total_visits
                )
            return score


    @staticmethod
    def opponents_score(score: float):
        assert 0 <= score <= 1
        return 1 - score



    ### Training and Backpropagation

    def single_run(self):
        if self.is_gameover:
            self.backpropagate(self.utility)
        elif not self.is_expanded:
            child = self.expand()
            score = child.simulate()
            child.backpropagate(score)
        else:
            # Recurse down tree, until a gameover or not expanded node is found
            action = self.get_exploration_action()
            child  = self.children[action]
            child.single_run()


    def expand(self) -> 'MCNode':
        assert not self.is_gameover
        assert not self.is_expanded

        unexpanded = self.get_unexpanded()
        assert len(unexpanded)

        action     = np.random.choice(unexpanded)
        child      = self.create_child(action)
        return child


    def simulate(self) -> float:
        return run_random_simulation(self.bitboard, self.player_id)


    def backpropagate(self, score: float):
        node = self
        while 1:
            node.total_score  += score
            node.total_visits += 1
            if node.parent is not None:
                node  = node.parent
                score = self.opponents_score(score)
            else:
                break  # Reached the root node, so terminate




root_nodes: List[Union[MCNode,None]] = [ None, None, None ]  # root_nodes[observation.mark]

def MontyCarloLinkedList(**kwargs):
    # observation   = {'mark': 1, 'board': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    # configuration = {'columns': 7, 'rows': 6, 'inarow': 4, 'steps': 1000, 'timeout': 8}
    def _MontyCarloLinkedList_(observation: Struct, _configuration_: Struct) -> int:
        first_move_time = 0
        safety_time     = 2
        start_time      = time.perf_counter()
        configuration   = cast_configuration(_configuration_)

        player_id     = int(observation.mark)
        listboard     = np.array(observation.board, dtype=np.int8)
        bitboard      = list_to_bitboard(listboard)
        move_number   = get_move_number(bitboard)
        is_first_move = int(move_number < 2)
        endtime       = start_time + _configuration_.timeout - safety_time - (first_move_time * is_first_move)

        root_node = root_nodes[player_id]
        if root_node is None or root_node.find_child(bitboard) is None:
            root_node = root_nodes[player_id] = MCNode(
                bitboard    = bitboard,
                player_id   = player_id,
                parent      = None,
                config      = configuration,
                exploration = kwargs.get('exploration', 1.0)
            )
        else:
            root_node = root_nodes[player_id] = root_nodes[player_id].find_child(bitboard)
        assert root_node is not None

        count = 0
        while time.perf_counter() < endtime:
            count += 1
            root_node.single_run()

        action     = root_node.get_best_action()
        time_taken = time.perf_counter() - start_time
        print(f'MontyCarloLinkedList: p{player_id} action = {action} after {count} simulations in {time_taken:.3f}s')
        return int(action)
    return _MontyCarloLinkedList_

def MontyCarloLinkedListKaggle(observation, _configuration_):
    return MontyCarloLinkedList()(observation, _configuration_)
