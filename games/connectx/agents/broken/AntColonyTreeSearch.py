# Ant Colony Tree Search is inspired by Monty Carl Tree Search
# However the choice of which node to expand is chosen via a random weighting of past scores
import time
from struct import Struct
from typing import Union

from numba import int32

from core.ConnectXBBNN import *
from core.ConnectXBBNN import configuration



configuration = configuration  # prevent optimize imports from removing configuration import

### Configuration

Hyperparameters = namedtuple('hyperparameters', ['pheromones_power', 'initial_pheromones'])
hyperparameters = Hyperparameters(
    pheromones_power   = 1.25,
    initial_pheromones = 16
)


### Pheromones

# NOTE: @njit here causes Unknown attribute: numba.types.containers.UniTuple
Pheromones = namedtuple('pheromones', ['rewards', 'totals'])
def new_pheromones()-> Pheromones:
    global configuration
    rewards = numba.typed.Dict.empty(
        key_type   = numba.types.containers.UniTuple(int64, 2),
        value_type = int32[:]
    )
    totals = numba.typed.Dict.empty(
        key_type   = numba.types.containers.UniTuple(int64, 2),
        value_type = int32[:]
    )

    # pheromone.rewards[bitboard.tobytes()][action]  += 1
    # pheromone.total[bitboard.tobytes()][action]    += 1
    pheromones = Pheromones(rewards=rewards, totals=totals)
    return pheromones

pheromones      = new_pheromones()
pheromones_type = numba.typeof(pheromones)
# print(pheromones_type)

@njit
def init_pheromones(pheromones: Pheromones, bitboard: np.ndarray) -> None:
    global hyperparameters
    hash = (bitboard[0], bitboard[1])
    if not hash in pheromones.rewards:
        initial = hyperparameters.initial_pheromones
        moves   = get_all_moves()
        totals  = np.array([ initial for _ in range(len(moves)) ], dtype=np.int32)
        rewards = np.array([ initial for _ in range(len(moves)) ], dtype=np.int32)
        pheromones.rewards[hash] = rewards
        pheromones.totals[hash]  = totals
    return None

@njit
def update_pheromones(pheromones: Pheromones, bitboard: np.ndarray, action: int, reward: int):
    init_pheromones(pheromones, bitboard)

    hash = (bitboard[0], bitboard[1])
    pheromones.rewards[hash][action] += reward
    pheromones.totals[hash][action]  += 2
    return None

### Simulate

@njit
def weighted_choice(options: np.ndarray, weights: np.ndarray) -> Union[int,float]:
    """Returns weighted choice from options array given unnormalized weights"""
    assert len(options) == len(weights) != 0
    total = np.sum(weights)
    rand  = np.random.rand() * total
    for i in range(len(options)):
        option = options[i]
        weight = weights[i]
        if weight < rand:
            rand -= weight
        else:
            break
    return option



@njit()
def get_weighted_move(bitboard: np.ndarray, pheromones: Pheromones) -> int:
    global hyperparameters
    assert not has_no_more_moves(bitboard), 'get_weighted_move() called when has_no_more_moves()'

    init_pheromones(pheromones, bitboard)

    moves   = get_legal_moves(bitboard)
    hash    = (bitboard[0], bitboard[1])
    rewards = pheromones.rewards[hash]                    # extract values array | pheromones.rewards[hash][action] = value
    totals  = pheromones.totals[hash]                     # extract values array | pheromones.rewards[hash][action] = value
    scores  = rewards / totals
    scores  = scores.astype(np.float64) ** hyperparameters.pheromones_power  # amplify sensitivity to pheromones
    weights = np.array([ scores[i] for i in moves ])
    action  = weighted_choice(moves, weights)
    return int(action)

@njit
def get_best_move(bitboard: np.ndarray, pheromones: pheromones_type) -> int:
    init_pheromones(pheromones, bitboard)
    hash    = (bitboard[0], bitboard[1])
    moves   = get_legal_moves(bitboard)
    rewards = pheromones.rewards[hash]
    totals  = pheromones.totals[hash]
    scores  = rewards / totals    # scores in valid move order
    scores  = np.array([ scores[i] for i in moves ])
    action  = moves[ int(np.argmax(scores)) ]
    return action


@njit
def simulate(bitboard: np.ndarray, pheromones: pheromones_type, original_player_id: int, player_id: int, depth: int = 0) -> int:
    if is_gameover(bitboard):
        return get_winner(bitboard)  # terminate recursion
    else:
        moves = get_legal_moves(bitboard)
        action = get_weighted_move(bitboard, pheromones)  # Prefer exploring more own successful moves

        next_bitboard  = result_action(bitboard, action, player_id)
        next_player_id = 3 - player_id  # swaps players 1<->2
        winner         = simulate(next_bitboard, pheromones, original_player_id, next_player_id, depth+1)
        reward         = 1 if winner == 0 else 2 if (winner == player_id) else 0

        # This is a recursive function, so pheromone updates will be backpropagated up the stack
        update_pheromones(pheromones, bitboard, action, reward)
        return winner



# The last function defined in the file run by Kaggle in submission.csv
# BUGFIX: duplicate top-level function names in submission.py can cause a Kaggle Submission Error
def AntColonyTreeSearch(observation: Struct, _configuration_: Struct) -> int:
    # observation   = {'mark': 1, 'board': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    # configuration = {'columns': 7, 'rows': 6, 'inarow': 4, 'steps': 1000, 'timeout': 8}

    safety_time = 1   # == 1000ms + 1s if first move (to allow numba to compile)
    start_time  = time.perf_counter()

    global configuration
    configuration = cast_configuration(_configuration_)

    global pheromones
    # pheromones    = new_pheromones()
    player_id     = observation.mark
    listboard     = numba.typed.List(observation.board)
    bitboard      = list_to_bitboard(listboard)

    # first_move  = int( np.count_nonzero(observation.board) >= 2 )
    first_move  = int( get_move_number(bitboard) < 2 )
    endtime     = start_time + _configuration_.timeout - safety_time - first_move
    simulations = 0
    while time.perf_counter() < endtime:
        simulations += 1
        simulate(bitboard, pheromones, player_id, player_id)

    action = get_best_move(bitboard, pheromones)
    # action   = get_weighted_move(bitboard, pheromones)
    if first_move: action = 3  # hardcode first move, but use time available to @njit compile and simulate pheromones
    time_taken = time.perf_counter() - start_time
    print(f'AntColonyTreeSearch: p{player_id} action = {action} after {simulations} simulations in {time_taken:.3f}s')
    return int(action)
