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
import sys
import time
from struct import Struct
from typing import Union

from numba import float32
from numba import typed

from core.ConnectXBBNN import *
from core.ConnectXBBNN import configuration
from util.weighted_choice import weighted_choice



### Configuration

configuration = configuration  # prevent removal from imports

class Hyperparameters(namedtuple('hyperparameters', ['exploration'])):
    exploration: float = np.sqrt(2)
hyperparameters = Hyperparameters(exploration=np.sqrt(2))



### State

# BUGFIX: Using an @njit function as a wrapper rather tha a namedtuple() fixes casting issues on Kaggle
@njit
def PathEdge(bitboard: np.ndarray, action: int) -> Tuple[np.ndarray, int]:
    return (bitboard, action)


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
def update_state( state: typed.Dict, bitboard: np.ndarray, action: int, wins: Union[int, float], totals: Union[int, float] ):
    hash = hash_bitboard(bitboard)
    if hash not in state:
        init_state(state, bitboard)
    state[hash][action] += np.array([wins, totals], dtype=np.float32)
    return state


### Selection

@njit
def path_selection( state: typed.Dict, bitboard: np.ndarray, player_id: int, simulation_count: int ) -> typed.List:  # List[Tuple[np.ndarray, int]]
    """
    Selection:
      leaf_node = root_node
      while leaf_node not expanded or gameover
        exploration = √2
        choose random lead_node with probability: wins/total + exploration * sqrt( ln(simulation_count)/total )
      return path (ending in player_id move)
    """
    path        = typed.List()
    actions     = get_all_moves()
    last_player = next_player = current_player_id(bitboard)


    while is_expanded(state, bitboard) and not is_gameover(bitboard):
        hash = hash_bitboard(bitboard)
        if hash not in state:
            break
        else:
            last_player = next_player
            weights     = get_child_probabilities(state, bitboard, simulation_count)
            action      = weighted_choice(actions, weights)
            path.append( PathEdge(bitboard, action) )

            # Update variables for next loop
            bitboard    = result_action(bitboard, action, next_player)
            next_player = next_player_id(next_player)
            assert next_player == current_player_id(bitboard)

    # Path Selection should always end with a node belonging to player_id
    if player_id != last_player:
        path.append( PathEdge(bitboard, -1) )  # Add unexpanded leaf node to end of path

    assert len(path) > 0
    # assert player_id == current_player_id(path[-1][0])
    return path


@njit
def get_child_probabilities( state: typed.Dict, bitboard: np.ndarray, simulation_count: int, normalize=True ):
    global hyperparameters

    hash       = hash_bitboard(bitboard)
    win_totals = state[hash]
    wins       = win_totals[:,0]
    totals     = win_totals[:,1]

    # Avoid divide by zero | Avoid log(1) == 0 | Avoid illegal moves
    totals[totals <= 0] = 1.0
    simulation_count    = max(simulation_count, 2)

    weights = wins/totals + hyperparameters.exploration * np.sqrt( np.log(simulation_count)/totals )

    # Set probability of any illegal moves to 0
    if not has_no_illegal_moves(bitboard):
        for action in get_all_moves():
        # for action in range(weights.shape[0]):
            if not is_legal_move(bitboard, action):
                weights[action] = 0

    # Normalize so probabilities sum to 1
    if normalize:
        weights = weights / np.sum(weights)

    # print('wins', wins, 'totals', totals, 'weights', weights.round(2))  # DEBUG
    return weights



### Expansion

@njit
def is_expanded( state: typed.Dict, bitboard: np.ndarray ):
    hash = hash_bitboard(bitboard)
    if hash not in state:
        return False

    win_totals = state[hash]
    unexpanded = (win_totals[:,1] == 0)
    if np.any(unexpanded):
        for action in np.where(unexpanded)[0]:
            if is_legal_move(bitboard, action):
                return False
    return True

@njit
def expand_node(state: typed.Dict, path: typed.List, bitboard: np.ndarray, player_id: int) -> int:
    """ Look for any child nodes that have not yet been expanded, and run one random simulation for each """
    # path: typed.List[Tuple[np.ndarray, int]]
    hash = hash_bitboard(bitboard)
    if hash not in state:
        init_state(state, bitboard)
    win_totals = state[hash]
    unexpanded = np.where(win_totals[:,1] == 0)[0]

    wins   = 0
    totals = 0
    for n in range(len(unexpanded)):
        action = unexpanded[n]
        if not is_legal_move(bitboard, action): continue

        result  = result_action(bitboard, action, player_id)
        score   = run_simulation(result, player_id)
        wins   += score
        totals += 1
        backpropergate_scores(state, [ PathEdge(bitboard, action) ], player_id, score)

    # Backpropergate combined wins/totals through parent path
    if totals > 0:
        backpropergate_scores(state, path, player_id, wins, totals)

    return totals  # returns number of simulations run


### Simulate

@njit
def run_simulation( bitboard: np.ndarray, player_id: int ) -> float:
    """ Returns +1 = victory | 0.5 = draw | 0 = loss """
    move_number = get_move_number(bitboard)
    next_player = 1 if move_number % 2 == 0 else 2  # player 1 has the first move on an empty board
    while not is_gameover(bitboard):
        action      = get_random_move(bitboard)
        bitboard    = result_action(bitboard, action, next_player)
        next_player = next_player_id(next_player)
        # print( bitboard_to_numpy2d(bitboard) )  # DEVUG
    score = get_utility_zero_one(bitboard, player_id)
    return score



### Backpropergate

@njit
def backpropergate_scores(state: typed.Dict, path: typed.List, player_id: int, score: Union[int,float], total: int = 1) -> None:
    # path: typed.List[Tuple[np.ndarray, int]]
    assert len(path) != 0
    assert score <= total

    current_player = 0
    for n in range(len(path)):     # numba prefers explict for loops
        node, action = path[n]
        if action <= -1: continue  # ignore unexpanded and leaf nodes | BUGFIX: numba typed.List([]) needs runtime type hint

        current_player = current_player or current_player_id(path[0][0])              # defer lookup until after action <= -1
        player_score   = score if (current_player == player_id) else (total - score)  # +1 = victory | 0.5 = draw | 0 = loss
        update_state( state, node, action, player_score, total )

        # Update variables for next loop
        player_id      = next_player_id(player_id)
    return None


#### Selection

# cannot @jit nopython=True required to access time.perf_counter()
def run_search(state: typed.Dict, bitboard: np.ndarray, player_id: int, endtime: float = 0, iterations = 0) -> Tuple[int,int]:
    """
    This is the main loop
    - Ensure root node is initialized and expanded
    - Repeat Monty Carlo TreeSearch until endtime
    - Return best action and simulation count
    """
    assert endtime or iterations

    init_state(state, bitboard)

    path_to_root_node = typed.List()
    path_to_root_node.append( PathEdge(bitboard, -1) )          # BUGFIX: numba typed.List([]) needs runtime type hint
    expand_node(state, path_to_root_node, bitboard, player_id)  # Ensure root node is always expanded

    count = 0
    if iterations:
        count = run_search_loop(state, bitboard, player_id, count, iterations)
    else:
        # NOTE: time.perf_counter() and raw python loops are slow compared to numba, so run most of loop inside #@njit
        while time.perf_counter() < endtime:
            count = run_search_loop(state, bitboard, player_id, count, 100)

    final_weights = get_child_probabilities(state, bitboard, count)
    action        = int(np.argmax(final_weights))  # actions are defined by index
    return action, count

@njit
def run_search_loop( state: typed.Dict, bitboard: np.ndarray, player_id: int, simulation_count: int = 1, iterations: int = 1 ) -> int:
    """
    Run N Monty Carlo Tree Search simulations
    - Start at the Root node
    - Perform Path Selection to find a most probable leaf_node to explore
    - Run Random Simulation from leaf node
    - Backpropergate Scores
    """
    actions = get_all_moves()
    iterations_end = simulation_count + iterations
    while simulation_count <= iterations_end:
        simulation_count += 1
        path              = path_selection(state, bitboard, player_id, simulation_count)
        leaf_bitboard, _  = path[-1]
        simulation_count += expand_node(state, path, leaf_bitboard, player_id)

        if not is_gameover(leaf_bitboard):
            weights           = get_child_probabilities(state, leaf_bitboard, simulation_count)
            action            = weighted_choice(actions, weights)
            leaf_bitboard     = result_action(leaf_bitboard, action, player_id)
            simulation_count += expand_node(state, path, leaf_bitboard, player_id)

        score = run_simulation(leaf_bitboard, player_id)
        backpropergate_scores(state, path, player_id, score)
    return simulation_count



### recompile Numba
### TODO: This needs to happen within 16 of first turn and 8s of second turn

def precompile_numba_lite(move_number: int):
    """ Only @jit compile a subset of functions on first turn = wait until turn 2+ to compile the rest"""
    if move_number // 2 == 1:
        state    = new_state()
        bitboard = result_action(empty_bitboard(), 3, 1)
        hash     = hash_bitboard(bitboard)
        winner   = get_winner(bitboard)


def precompile_numba():
    time_start = time.perf_counter()
    state = new_state()
    run_search(state, empty_bitboard(), player_id=1, endtime=sys.maxsize, iterations=1)
    time_taken = time.perf_counter() - time_start
    print(f'precompile_numba() in {time_taken:0.2f}s')


### Main

state = new_state()  # Shared State

# precompile_numba()  # kaggle offers 16s to make the first move - but we don't get extra time by running this outside the agent

# The last function defined in the file run by Kaggle in submission.csv
# BUGFIX: duplicate top-level function names in submission.py can cause a Kaggle Submission Error
def MontyCarloTreeSearch(observation: Struct, _configuration_: Struct) -> int:
    # observation   = {'mark': 1, 'board': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    # configuration = {'columns': 7, 'rows': 6, 'inarow': 4, 'steps': 1000, 'timeout': 8}

    first_move_time = 1
    safety_time     = 0.25    # Only gets checked once every hundred simulations
    start_time      = time.perf_counter()

    global configuration
    configuration = cast_configuration(_configuration_)

    global state  # Share state between runs
    # state = new_state()

    player_id     = observation.mark
    listboard     = np.array(observation.board, dtype=np.int8)
    bitboard      = list_to_bitboard(listboard)
    move_number   = get_move_number(bitboard)
    is_first_move = int(move_number < 2)
    endtime       = start_time + _configuration_.timeout - safety_time - (first_move_time * is_first_move)

    # if move_number <= 2:
    #     # Kaggle gives us a 16s window on first move to @jit compile, so only compile core functions on first move
    #     precompile_numba_lite(move_number)
    #     return 3

    action, count = run_search(state, bitboard, player_id, endtime)

    # if is_first_move: action = 3  # hardcode first move, but use time available to #@njit compile and simulate state
    time_taken = time.perf_counter() - start_time
    print(f'MontyCarloTreeSearch: p{player_id} action = {action} after {count} simulations in {time_taken:.3f}s')

    return int(action)          # kaggle_environments requires a python int, not np.int32
