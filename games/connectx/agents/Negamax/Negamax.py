
# The last function defined in the file run by Kaggle in submission.csv
# BUGFIX: duplicate top-level function names in submission.py can cause a Kaggle Submission Error
import time
from struct import Struct

from heuristics.BitboardHeuristic import *



actions = get_all_moves()


# @njit
def negamax( bitboard: np.ndarray, player_id: int, depth: int, max_count=0, alpha=-np.inf, beta=np.inf, action=0, count=0) -> Tuple[int,float,int]:
    """
    DOCS: https://en.wikipedia.org/wiki/Negamax
    function negamax(node, depth, α, β, color) is
        if depth = 0 or node is a terminal node then
            return color × the heuristic value of node

        childNodes := generateMoves(node)
        childNodes := orderMoves(childNodes)
        value := −∞
        foreach child in childNodes do
            value := max(value, −negamax(child, depth − 1, −β, −α, −color))
            α := max(α, value)
            if α ≥ β then
                break (* cut-off *)
        return value
    """
    if is_gameover(bitboard):
        best_score  = get_utility_inf(bitboard, player_id)
        best_action = action
    elif depth == 0:
        best_score  = bitboard_gameovers_heuristic(bitboard, player_id)
        best_action = action
    else:
        next_player  = 3 - player_id  # 1 if player_id == 2 else 2
        actions      = get_legal_moves(bitboard)
        best_action  = np.random.choice(actions)
        best_score   = -np.inf
        scores       = []  # for debugging
        for n in range(len(actions)):
            action      = actions[n]
            child_node  = result_action(bitboard, action, player_id)
            child_action, child_score, count = negamax(
                bitboard  = child_node,
                player_id = next_player,
                depth     = depth-1,
                max_count = max_count,
                alpha     = -beta,
                beta      = -alpha,
                action    = action,
                count     = count+1
            )
            child_score = -child_score  # Negamax
            scores.append(child_score)  # for debugging

            if child_score > best_score:
                best_action = action
                best_score  = child_score
            if child_score > alpha:
                alpha       = child_score
            if alpha >= beta:
                break
            if max_count and count >= max_count:
                break

    return best_action, best_score, count



def negamax_deepening( bitboard: np.ndarray, player_id: int, min_depth=1, max_depth=10, depth_step=1, timeout=0.0, verbose_depth=True ) -> int:
    # The real trick with iterative deepening is caching, which allows us to out-depth the default minimax Agent
    if verbose_depth: print('\n'+ 'Negamax'.ljust(23) +' | depth:', end=' ', flush=True)

    time_start  = time.perf_counter()
    best_action = np.random.choice(get_legal_moves(bitboard))
    best_score  = -np.inf
    max_count   = 0
    count       = 0
    for depth in range(min_depth, max_depth+1, depth_step):
        action, score, count = negamax(
            bitboard  = bitboard,
            player_id = player_id,
            max_count = max_count,
            depth     = depth,
            count     = count
        )
        time_taken     = time.perf_counter() - time_start
        time_per_count = time_taken / count
        max_count      = timeout    / time_per_count // 1

        if verbose_depth: print(depth, end=' ', flush=True)
        if max_count and count >= max_count:
            break
        else:
            best_action = action
            best_score  = score
        if abs(best_score) == np.inf:
            if verbose_depth: print(best_score, end=' ', flush=True)
            break  # terminate iterative deepening on inescapable victory condition


        # Update variables for next loop
        # print(f'time_taken = {time_taken:.2f} | time_per_count = {time_per_count:.8f} | max_count = {max_count:.0f} | count = {count}')

    time_taken = time.perf_counter() - time_start
    if verbose_depth:
        print(f' = {best_action} ({best_score:.2f}) | {count} iterations in {time_taken:4.2f}s ', flush=True)
    # print(f'Negamax: p{player_id} depth {depth} = action {action} (score {score:.2f}) | {count} iterations in {time_taken:.3f}s')
    return int(best_action)


def precompile_numba():
    negamax(bitboard = empty_bitboard(), player_id = 1, depth = 1)
precompile_numba()

def Negamax():
    def _Negamax(observation: Struct, _configuration_: Struct) -> int:
        # observation   = {'mark': 1, 'board': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
        # configuration = {'columns': 7, 'rows': 6, 'inarow': 4, 'steps': 1000, 'timeout': 8}

        first_move_time = 0
        safety_time     = 2    # Only gets checked once every hundred simulations
        start_time      = time.perf_counter()

        global configuration
        configuration = cast_configuration(_configuration_)
        player_id     = observation.mark
        listboard     = np.array(observation.board, dtype=np.int8)
        bitboard      = list_to_bitboard(listboard)
        move_number   = get_move_number(bitboard)
        is_first_move = int(move_number < 2)
        timeout       = _configuration_.timeout - safety_time - (first_move_time * is_first_move) - (time.perf_counter() - start_time)
        print(time.perf_counter() - start_time)
        action = negamax_deepening(
            bitboard   = bitboard,
            player_id  = player_id,
            min_depth  = 3,
            max_depth  = 100,
            depth_step = 1,
            timeout    = timeout,
            verbose_depth = True
        )
        return int(action)  # kaggle_environments requires a python int, not np.int32
    return _Negamax
