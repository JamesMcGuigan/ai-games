
# The last function defined in the file run by Kaggle in submission.csv
# BUGFIX: duplicate top-level function names in submission.py can cause a Kaggle Submission Error
import time
from struct import Struct

from heuristics.BitboardGameoversHeuristic import *

configuration = configuration  # prevent optimize imports from removing configuration import

# @njit
def negamax(
        bitboard: np.ndarray,
        player_id:  int,
        depth:      int,
        max_count:  int   =  0,
        alpha:      float = -np.inf,
        beta:       float =  np.inf,
        action:     int   =  0,
        count:      int   =  0,
        sort_scores = np.zeros((configuration.columns,), dtype=np.float),
        recursing   = False
    ) -> Tuple[int,float,int, np.ndarray]:
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
    best_action  = action  # default action incase of gameover or depth=0
    scores       = np.zeros((configuration.columns,), dtype=np.float32)

    if is_gameover(bitboard):
        best_score  = get_utility_inf(bitboard, player_id)
    elif depth == 0:
        best_score = bitboard_gameovers_heuristic(bitboard, player_id)
    else:
        actions = get_legal_moves(bitboard)
        actions = sorted(actions, key=lambda action: sort_scores[action], reverse=True)   # try the best parent moves first
        if not recursing:
            actions = [ action for action in actions if sort_scores[action] != -np.inf ]  # auto-exclude losing moves from negamax_deepening()
            scores[ sort_scores == -np.inf ] = -np.inf                                    # persist -np.inf scores
            if len(actions) == 0:
                # All moves are losing, pick the best one by heuristic
                return negamax(bitboard, player_id, depth=1)
        if len(actions) == 1:
            # Short circuit as we have only one valid move
            best_action = actions[0]
            best_score  = bitboard_gameovers_heuristic(bitboard, player_id)
        else:
            # Recurse until we reach maximum depth
            next_player = 3 - player_id  # 1 if player_id == 2 else 2
            best_action  = actions[0]
            best_score   = -np.inf
            child_scores = sort_scores
            for n in range(len(actions)):
                next_action = actions[n]
                child_node  = result_action(bitboard, next_action, player_id)
                child_action, child_score, count, child_scores = negamax(
                    bitboard    = child_node,
                    player_id   = next_player,
                    depth       = depth-1,
                    max_count   = max_count,
                    alpha       = -beta,
                    beta        = -alpha,
                    action      = next_action,
                    count       = count+1,
                    sort_scores = child_scores,
                    recursing   = True
                )
                action_score = -child_score  # Negamax
                scores[next_action] = action_score

                if action_score > best_score:
                    best_action = next_action
                    best_score  = action_score
                    if best_score == np.inf:
                        break             # we have found a winning move!
                if max_count and count >= max_count:
                    break                 # Deterministic timeout

                # BUGFIX: disable AlphaBeta pruning as heuristic is inadmissible and fails foresee future double attacks
                # if action_score > alpha:
                #     alpha = action_score  # AlphaBeta pruning - Keep track of best score that other paths have to beat
                # if alpha >= beta:
                #     break                 # AlphaBeta pruning - this path is worse than others in the tree, so ignore

    scores[best_action] = best_score
    return best_action, best_score, count, scores



def negamax_deepening(
        bitboard: np.ndarray,
        player_id: int,
        min_depth  = 1,
        max_depth  = 10,
        depth_step = 1,
        timeout    = 0.0,
        verbose    = 1
) -> int:
    # The real trick with iterative deepening is caching, which allows us to out-depth the default minimax Agent
    if verbose: print('\n'+ 'Negamax'.ljust(23) +' | depth:', end=' ', flush=True)

    time_start  = time.perf_counter()
    legal_moves = get_legal_moves(bitboard)
    best_action = np.random.choice(legal_moves)
    best_score  = -np.inf
    max_count   = 0
    count       = 0
    depth       = 0
    scores      = np.zeros((configuration.columns,), dtype=np.float)
    for depth in range(min_depth, max_depth+1, depth_step):
        action, score, count, scores = negamax(
            bitboard    = bitboard,
            player_id   = player_id,
            max_count   = max_count,
            depth       = depth,
            count       = count,
            sort_scores = scores
        )
        if max_count and count >= max_count and count > 1000:
            break  # if we didn't complete a full depth search (after warmup) then ignore the results
        else:
            time_taken     = time.perf_counter() - time_start
            time_per_count = time_taken / (count + 1)           # BUGFIX: avoid divide by zero
            max_count      = timeout    / time_per_count // 1

        # Search results are valid now, so persist as candidate return values
        best_action    = action
        best_score     = score

        if verbose:
            print(depth, end=' ', flush=True)
        if verbose >= 2:
            print(f'Negamax: p{player_id} depth {depth} = action {best_action} (score {best_score:.2f}) | {count} iterations in {time_taken:.3f}s')
            print('scores', scores.round(2).tolist(),'\n')

        if abs(best_score) == np.inf:
            if verbose: print(best_score, end=' ', flush=True)
            break  # terminate iterative deepening on inescapable victory condition

        non_losing_moves = [ action for action in legal_moves if scores[action] != -np.inf ]
        if len(non_losing_moves) <= 1:
            break  # We either have one remaining move, or else all moves are losing - no need to continue searching

    time_taken = time.perf_counter() - time_start
    if verbose:
        print(f' = {best_action} ({best_score:.2f}) | {count} iterations in {time_taken:4.2f}s ', flush=True)

    return int(best_action)


def precompile_numba():
    negamax(bitboard = empty_bitboard(), player_id = 1, depth = 1)

def Negamax(**kwargs):
    def _Negamax_(observation: Struct, _configuration_: Struct) -> int:
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
        if is_first_move:
            precompile_numba()

        timeout = _configuration_.timeout - safety_time - (first_move_time * is_first_move) - (time.perf_counter() - start_time)
        action  = negamax_deepening(
            bitboard   = bitboard,
            player_id  = player_id,
            min_depth  = kwargs.get('min_depth',  1),
            max_depth  = kwargs.get('max_depth',  100),
            depth_step = kwargs.get('depth_step', 1),
            timeout    = kwargs.get('timeout',    timeout),
            verbose    = kwargs.get('verbose',    1)
        )
        if is_first_move:
            return 3            # BUGFIX: always play the center on the first move | TODO: create opening book
        else:
            return int(action)  # kaggle_environments requires a python int, not np.int32
    return _Negamax_

def NegamaxKaggle(observation: Struct, _configuration_: Struct) -> int:
    return Negamax()(observation, _configuration_)
