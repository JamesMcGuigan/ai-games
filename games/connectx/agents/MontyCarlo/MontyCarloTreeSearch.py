# Monty Carlo Tree Search
# Video: https://www.youtube.com/watch?v=Fbs4lnGLS8M
# Wiki:  https://en.wikipedia.org/wiki/Monte_Carlo_tree_search#Monte_Carlo_Method
#
# Basic Algorithm:
# Loop until timeout:
#   Selection:
#     start at root node
#     while node not expanded or gameover:
#       exploration = √2
#       choose random child with probability: wins/total + exploration * sqrt( ln(simulation_count)/total )
#   Expansion: run one end-to-end simulation for each child node
#   Update:    for each parent node, update win/total statistics
import time
from struct import Struct
from typing import Union

from numba import float32

import core.ConnectXBBNN
from core.ConnectXBBNN import *
from util.weighted_choice import weighted_choice



### Configuration

configuration = core.ConnectXBBNN.configuration

class Hyperparameters(namedtuple('hyperparameters', ['exploration'])):
    exploration: float = np.sqrt(2)
hyperparameters = Hyperparameters(exploration=np.sqrt(2))



### State

class PathEdge(namedtuple('PathEdge', ['bitboard', 'action'])):
    bitboard: np.ndarray
    action:   int


# NOTE: @njit here causes Unknown attribute: numba.types.containers.UniTuple
def new_state()-> numba.typed.Dict:
    global configuration
    state = numba.typed.Dict.empty(
        key_type   = numba.types.containers.UniTuple(int64, 2),  # tuple(min(bitboard, reverse_bits))
        value_type = float32[:,:]                                # state[hash][action] = [wins, total]
    )
    return state

@njit
def init_state( state: typed.Dict, bitboard: np.ndarray ) -> None:
    global hyperparameters
    global configuration
    hash = hash_bitboard(bitboard)
    if not hash in state:
        state[hash] = np.zeros((configuration.columns, 2), dtype=np.float32)  # state[hash][action] = [wins, total]
    return None

@njit
def update_state( state: typed.Dict, bitboard: np.ndarray, action: int, reward: Union[int,float] ):
    hash = hash_bitboard(bitboard)
    if hash not in state:
        init_state(state, bitboard)
    state[hash][action] += np.array([reward, 1.0], dtype=np.float32)
    return state


### Selection

@njit
def path_selection( state: typed.Dict, bitboard: np.ndarray, simulation_count: int ) -> typed.List:  # List[Tuple[np.ndarray, int]]
    """
    Selection:
        start at root node
        while node not expanded or gameover:
            exploration = √2
            choose random child with probability: wins/total + exploration * sqrt( ln(simulation_count)/total )
    """
    path       = typed.List()
    actions    = get_all_moves()
    current_player = current_player_id(bitboard)
    while is_expanded(state, bitboard) and not is_gameover(bitboard):
        hash = hash_bitboard(bitboard)
        if hash not in state:
            break
        else:
            last_bitboard  = bitboard
            weights        = get_child_probabilities(state, bitboard, simulation_count)
            action         = weighted_choice(actions, weights)
            bitboard       = result_action(bitboard, action, current_player)
            current_player = next_player_id(current_player)
            path.append(PathEdge(last_bitboard, action))  # backpropergate() requires starting_board and next_move
    return path


@njit
def get_child_probabilities( state: typed.Dict, bitboard: np.ndarray, simulation_count: int, normalize=False ):
    global hyperparameters
    if simulation_count == 0: simulation_count += 1  # avoid divide by zero

    hash       = hash_bitboard(bitboard)
    win_totals = state[hash]
    weights    = np.zeros((win_totals.shape[0],), dtype=np.float32)
    for action in range(win_totals.shape[0]):
        if not is_legal_move(bitboard, action):
            score = 0
        else:
            wins  = win_totals[action][0]
            total = win_totals[action][1]
            score = wins/total + hyperparameters.exploration * np.sqrt( np.log(simulation_count)/total )
        weights[action] = score
    if normalize:
        weights = weights / np.sum(weights)
    return weights



### Expansion

@njit
def is_expanded( state: typed.Dict, bitboard: np.ndarray ):
    hash = hash_bitboard(bitboard)
    if hash not in state: return False

    win_totals = state[hash]
    actions    = get_all_moves()
    for action in range(len(actions)):
        totals = win_totals[action][1]
        if totals == 0 and is_legal_move(bitboard, action):
            return False
    return True

@njit
def expand_node(state: typed.Dict, path: typed.List, bitboard: np.ndarray, player_id: int) -> None:
    # path: typed.List[Tuple[np.ndarray, int]]
    hash        = hash_bitboard(bitboard)
    win_totals  = state[hash]
    actions     = get_all_moves()
    for action in range(len(actions)):
        totals = win_totals[action][1]
        if totals == 0 and is_legal_move(bitboard, action):
            result    = result_action(bitboard, action, player_id)
            score     = run_simulation(result, player_id)

            backpropergate_scores(state, path, player_id, score)
            backpropergate_scores(state, [ PathEdge(bitboard, action) ], player_id, score)

    return None


### Simulate

@njit
def run_simulation( bitboard: np.ndarray, player_id: int ) -> float:
    """ Returns +1 = victory | 0.5 = draw | 0 = loss """
    move_number = get_move_number(bitboard)
    next_player = 1 if move_number % 2 == 0 else 2  # player 1 has the first move on an empty board
    while not is_gameover(bitboard):
        action      = get_random_move(bitboard)
        bitboard    = result_action(bitboard, action, next_player)
        next_player = next_player_id(player_id)
    return get_utility_zero_one(bitboard, player_id)



### Backpropergate

@njit
def backpropergate_scores(state: typed.Dict, path: typed.List, player_id: int, score: Union[int,float]) -> None:
    # path: typed.List[Tuple[np.ndarray, int]]
    assert len(path) != 0
    current_player = current_player_id(path[0][0])
    for node, action in path:
        if action <= -1: continue  # BUGFIX: numba typed.List([]) needs a dummy value for typing
        player_score = score if current_player == player_id else 1 - score  # +1 = victory | 0.5 = draw | 0 = loss
        update_state( state, node, action, player_score )
        player_id = next_player_id(player_id)
    return None


#### Selection

# cannot @jit nopython=True required to access time.perf_counter()
def run_search(state: typed.Dict, bitboard: np.ndarray, player_id: int, endtime: float, iterations=0) -> Tuple[int,int]:

    init_state(state, bitboard)
    path_to_root_node = typed.List([ PathEdge(bitboard, -1) ])  # BUGFIX: numba typed.List([]) needs a dummy value for typing
    expand_node(state, path_to_root_node, bitboard, player_id)  # Ensure root node is always expanded

    actions = get_all_moves()

    count = 0
    if iterations:
        count = run_search_loop(state, bitboard, player_id, count, actions, iterations)
    else:
        # This is the main loop and time.perf_counter() is slow compared to numba, so run tight loop in @njit
        while time.perf_counter() < endtime:
            count = run_search_loop(state, bitboard, player_id, count, actions, 100)

    final_weights = get_child_probabilities(state, bitboard, count)
    action        = actions[ np.argmax(final_weights) ]
    return action, count

@njit
def run_search_loop( state: typed.Dict, bitboard: np.ndarray, player_id: int, count: int = 1, actions=get_all_moves(), iterations: int = 1 ) -> int:
    for count in range(count+1, count+iterations+1):
        path              = path_selection(state, bitboard, count)
        leaf_node, action = path[-1]

        expand_node(state, path, leaf_node, player_id)

        weights       = get_child_probabilities(state, leaf_node, count)
        action        = weighted_choice(actions, weights)
        next_bitboard = result_action(bitboard, action, player_id)
        score         = run_simulation(next_bitboard, player_id)

        backpropergate_scores(state, path, player_id, score)
    return count




### Main

# def precompile_numba(state):
#     run_search(state, empty_bitboard(), player_id=1, endtime=sys.maxsize, iterations=1)
#
# state = new_state()
# precompile_numba(state)

# The last function defined in the file run by Kaggle in submission.csv
# BUGFIX: duplicate top-level function names in submission.py can cause a Kaggle Submission Error
def MontyCarloTreeSearch(observation: Struct, _configuration_: Struct) -> int:
    # observation   = {'mark': 1, 'board': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    # configuration = {'columns': 7, 'rows': 6, 'inarow': 4, 'steps': 1000, 'timeout': 8}

    first_move_time = 0     # Numba is now precompiled
    safety_time     = 0.25  # Only gets checked once every hundred loops
    start_time      = time.perf_counter()


    global configuration
    configuration = cast_configuration(_configuration_)

    # global state  # Share state between runs
    state = new_state()

    player_id     = observation.mark
    listboard     = typed.List(); [ listboard.append(cell) for cell in observation.board ]  # BUGFIX: constructor fails to load data
    bitboard      = list_to_bitboard(listboard)
    move_number   = get_move_number(bitboard)
    is_first_move = int(move_number < 2)
    endtime       = start_time + _configuration_.timeout - safety_time - (first_move_time * is_first_move)


    action, count = run_search(state, bitboard, player_id, endtime)

    # if is_first_move: action = 3  # hardcode first move, but use time available to @njit compile and simulate state
    time_taken = time.perf_counter() - start_time
    print(f'MontyCarloTreeSearch: p{player_id} action = {action} after {count} simulations in {time_taken:.3f}s')
    return int(action)  # kaggle_environments requires a python int, not np.int32
