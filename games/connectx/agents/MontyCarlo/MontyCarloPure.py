# This is a LinkedList implementation of MontyCarlo Tree Search
# Inspired by https://www.kaggle.com/matant/monte-carlo-tree-search-connectx
import atexit
import time
from struct import Struct
from typing import Callable

from core.ConnectXBBNN import *
from util.base64_file import base64_file_load
from util.base64_file import base64_file_save

Hyperparameters = namedtuple('hyperparameters', [])

class MontyCarloNode:
    persist   = True
    save_node = {}                                                        # save_node[cls.__name__] = cls(empty_bitboard(), 1)
    root_nodes: List[Union['MontyCarloNode', None]] = [None, None, None]  # root_nodes[observation.mark]

    def __new__(cls, *args, **kwargs):
        # noinspection PyUnresolvedReferences
        # create a new cls.save_node and cls.root_nodes for each class
        for parentclass in cls.__mro__:  # https://stackoverflow.com/questions/2611892/how-to-get-the-parents-of-a-python-class
            if cls is parentclass: continue
            if ( cls.save_node  is getattr(parentclass, 'save_node',  None)
              or cls.root_nodes is getattr(parentclass, 'root_nodes', None)
            ):
                cls.cache      = {}
                cls.root_nodes = [None, None, None]
                break
        instance = object.__new__(cls)
        return instance

    def __init__(
            self,
            bitboard:      np.ndarray,
            player_id:     int,
            parent:        Union['MontyCarloNode', None] = None,
            parent_action: Union[int,None]       = None,
            exploration:   float = 1.0,
            **kwargs
    ):
        self.bitboard      = bitboard
        self.player_id     = player_id
        self.next_player   = 3 - player_id

        self.exploration   = exploration
        self.kwargs        = kwargs

        # self.mirror_hash   = hash_bitboard(bitboard)  # BUG: using mirror hashes causes get_best_action() to return invalid moves
        self.legal_moves   = get_legal_moves(bitboard)
        self.is_gameover   = is_gameover(bitboard)
        self.winner        = get_winner(bitboard) if self.is_gameover else 0
        self.utility       = 1 if self.winner == self.player_id else 0  # Scores in range 0-1

        self.parent        = parent
        self.parent_action = parent_action
        self.is_expanded   = False
        self.children: List[Union[MontyCarloNode, None]] = [None for action in get_all_moves()]  # include illegal moves to preserve indexing
        self.total_score   = 0.0
        self.total_visits  = 0
        self.ucb1_score    = self.get_ucb1_score(self)



    def __hash__(self):
        return tuple(self.bitboard)
        # return self.mirror_hash  # BUG: using mirror hashes causes get_best_action() to return invalid moves



    ### Loading and Saving

    @classmethod
    def prune(cls, node: 'MontyCarloNode', min_visits=7, pruned_count=0, total_count=0):
        for n, child in enumerate(node.children):
            if child is None: continue
            if child.total_visits < min_visits:
                pruned_count    += child.total_visits  # excepting terminal states, this equals the number of grandchildren
                total_count     += child.total_visits  # excepting terminal states, this equals the number of grandchildren
                node.children[n] = None
                node.is_expanded = False  # Use def expand(self) to reinitalize state
            else:
                total_count += 1
                pruned_count, total_count = cls.prune(child, min_visits, pruned_count, total_count)
        return pruned_count, total_count


    @classmethod
    def filename(cls):
        return f"data/{cls.__name__}_base64.py"

    @classmethod
    def load(cls):
        filename = cls.filename()
        loaded   = base64_file_load(filename)
        if loaded is not None:
            cls.save_node[cls.__name__] = loaded
            return loaded
        else:
            return None


    @classmethod
    def save(cls) -> Union[str,None]:
        if cls.persist == True and cls.save_node.get(cls.__name__, None) is not None:
            save_node    = cls.save_node[cls.__name__]

            start_time   = time.perf_counter()
            pruned_count, total_count = cls.prune(save_node)  # This reduces a 47MB base64 file down to 5Mb
            print(f'{cls.__name__}.save() - pruned {pruned_count:.0f}/{total_count:.0f} nodes '
                  + f'leaving {total_count-pruned_count:.0f} in {time.perf_counter() - start_time:.2f}s - ')

            filename = cls.filename()
            filesize = base64_file_save(save_node, filename)
            return filename
        return None



    ### Constructors and Lookups

    def create_child( self, action: int ) -> 'MontyCarloNode':
        result = result_action(self.bitboard, action, self.player_id)
        child  = None  # self.find_mirror_child(result, depth=1)  # BUG: using mirror hashes causes get_best_action() to return invalid moves
        if child is None:
            child = self.__class__(
                bitboard      = result,
                player_id     = next_player_id(self.player_id),
                parent        = self,
                parent_action = action,
                exploration   = self.exploration,
                **self.kwargs
            )
        self.children[action] = child
        self.is_expanded      = self._is_expanded()
        if self.is_expanded:
            self.on_expanded()
        return child


    def find_child( self, bitboard: np.array, depth=2 ) -> Union['MontyCarloNode', None]:
        assert 0 <= depth <= 2

        if depth >= 0:
            if np.all( self.bitboard == bitboard ):
                return self
        if depth >= 1:
            for child in self.children:
                if child is None: continue
                if np.all( child.bitboard == bitboard ):
                    return child
        if depth >= 2:
            # Avoid recursion to prevent duplicate calls to hash_bitboard()
            for child in self.children:
                if child is None: continue
                for grandchild in child.children:
                    if grandchild is None: continue
                    if np.all( grandchild.bitboard == bitboard ):
                        return grandchild
        return None

    # # BUG: using mirror hashes causes get_best_action() to return invalid moves
    # def find_mirror_child( self, bitboard: np.array, depth=2 ) -> Union['MontyCarloNode', None]:
    #     assert 0 <= depth <= 2
    #     mirror_hash = hash_bitboard(bitboard)
    #
    #     if depth >= 0:
    #         if self.mirror_hash == mirror_hash:
    #             return self
    #     if depth >= 1:
    #         for child in self.children:
    #             if child is None: continue
    #             if child.mirror_hash == mirror_hash:
    #                 return child
    #     if depth >= 2:
    #         # Avoid recursion to prevent duplicate calls to hash_bitboard()
    #         for child in self.children:
    #             if child is None: continue
    #             for grandchild in child.children:
    #                 if grandchild is None: continue
    #                 if grandchild.mirror_hash == mirror_hash:
    #                     return grandchild
    #     return None



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
        scores = [
            self.children[action].total_score
            if self.children[action] is not None else 0
            for action in self.legal_moves
        ]
        index  = np.argmax(scores)
        action = self.legal_moves[index]
        return action


    def get_exploration_action(self) -> int:
        scores = [
            self.children[action].ucb1_score
            if self.children[action] is not None else 0
            for action in self.legal_moves
        ]
        index  = np.argmax(scores)
        action = self.legal_moves[index]
        return action



    ### Scores

    def get_ucb1_score(self, node: 'MontyCarloNode') -> float:
        if node is None or node.total_visits == 0:
            return np.inf
        else:
            score = node.total_score / node.total_visits
            if node.parent is not None and node.parent.total_visits > 0:
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
            score = child.simulate()    # score from the perspective of the other player
            child.backpropagate(score)
        else:
            # Recurse down tree, until a gameover or not expanded node is found
            action = self.get_exploration_action()
            child  = self.children[action]
            child.single_run()


    def expand(self) -> 'MontyCarloNode':
        assert not self.is_gameover
        assert not self.is_expanded

        unexpanded = self.get_unexpanded()
        assert len(unexpanded)

        action     = np.random.choice(unexpanded)
        child      = self.create_child(action)
        return child

    def on_expanded(self) -> None:
        # Callback placeholder for any subclass hooks
        pass

    def simulate(self) -> float:
        return run_random_simulation(self.bitboard, self.player_id)


    def backpropagate(self, score: float):
        # child.simulate()  returns score for the player 2
        # child.total_score is accessed via the parent node, so score on this node is from the perspective of player 1
        node = self
        while node is not None:
            score = self.opponents_score(score)
            node.total_score  += score
            node.total_visits += 1
            node = node.parent      # when we reach the root: node.parent == None which terminates

        # get_ucb1_score() gets called 4x less often if we cache the value after backpropagation
        # get_ucb1_score() depends on parent.total_visits, so needs to be called after updating scores
        node = self
        while node is not None:
            node.ucb1_score = node.get_ucb1_score(node)
            node = node.parent      # when we reach the root: node.parent == None which terminates



    ### Agent
    @classmethod
    def agent(cls, **kwargs) -> Callable[[Struct, Struct],int]:
        def kaggle_agent( observation: Struct, _configuration_: Struct ):
            first_move_time = 0
            safety_time     = kwargs.get('safety_time', 0.25)
            start_time      = time.perf_counter()
            # configuration   = cast_configuration(_configuration_)

            player_id     = int(observation.mark)
            listboard     = np.array(observation.board, dtype=np.int8)
            bitboard      = list_to_bitboard(listboard)
            move_number   = get_move_number(bitboard)
            is_first_move = int(move_number < 2)
            endtime       = start_time + _configuration_.timeout - safety_time - (first_move_time * is_first_move)

            if cls.persist == True and cls.save_node.get(cls.__name__, None) is None:
                atexit.register(cls.save)
                cls.save_node[cls.__name__] = cls.load() or cls(empty_bitboard(), player_id=1)
                cls.root_nodes[1] = cls.root_nodes[2] = cls.save_node[cls.__name__]  # implement shared state

            root_node = cls.root_nodes[player_id]
            if root_node is None or root_node.find_child(bitboard, depth=2) is None:
                root_node = cls.root_nodes[player_id] = cls(
                    bitboard      = bitboard,
                    player_id     = player_id,
                    parent        = None,
                    # configuration = configuration,
                    **kwargs
                )
            else:
                root_node = cls.root_nodes[player_id] = cls.root_nodes[player_id].find_child(bitboard)
            assert root_node is not None

            count = 0
            while time.perf_counter() < endtime:
                count += 1
                root_node.single_run()

            action     = root_node.get_best_action()
            time_taken = time.perf_counter() - start_time
            print(f'{cls.__name__}: p{player_id} action = {action} after {count} simulations in {time_taken:.3f}s')
            return int(action)

        kaggle_agent.__name__ = cls.__name__
        return kaggle_agent

def MontyCarloPure(**kwargs):
    # observation   = {'mark': 1, 'board': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    # configuration = {'columns': 7, 'rows': 6, 'inarow': 4, 'steps': 1000, 'timeout': 8}
    def MontyCarloPure(observation: Struct, configuration: Struct) -> int:
        return MontyCarloNode.agent(**kwargs)(observation, configuration)
    return MontyCarloPure

def MontyCarloPureKaggle(observation, configuration):
    return MontyCarloPure()(observation, configuration)
