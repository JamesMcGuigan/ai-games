import math
import sys

from core.ConnectXBBNN import *



# Hyperparameters
single_square_score = 0.1  # Mostly ignore single squares, that can make lines in 8 directions
double_attack_score = 0.5  # 0.5 == 100% winrate vs AlphaBetaAgent

min_score =  math.inf  # min_score: -32.3
max_score = -math.inf  # max_score:  26.4

# Profiler (Macbook Pro 2011): This vectorized implementation is actually slower than unvectorized
# bitboard_gameovers_heuristic	                call_count=9469	    time=1558	own_time=1141
# bitboard_gameovers_heuristic_unvectorized	    call_count=9469	    time=1879	own_time=1862  ( 20% slower)
# bitboard_gameovers_heuristic_slow	            call_count=9469	    time=3419	own_time=1893  (220% slower)
@njit
def bitboard_gameovers_heuristic( bitboard: np.ndarray, player_id: int, gameovers: np.ndarray = get_gameovers() ) -> float:
    """ For all possible connect4 gameover positions,
        check if a player has at least one token in position and that the opponent is not blocking
        return difference in score

        Winrates:
         55% vs AlphaBetaAgent - original heuristic
         70% vs AlphaBetaAgent - + np.log2(p1_can_play) % 1 == 0
         60% vs AlphaBetaAgent - + double_attack_score=1   without np.log2() (mostly draws)
         80% vs AlphaBetaAgent - + double_attack_score=1   with np.log2() (mostly wins)
         80% vs AlphaBetaAgent - + double_attack_score=2   with np.log2() (mostly wins)
         70% vs AlphaBetaAgent - + double_attack_score=4   with np.log2() (mostly wins)
         80% vs AlphaBetaAgent - + double_attack_score=8   with np.log2() (mostly wins)
        100% vs AlphaBetaAgent - + double_attack_score=0.5 with np.log2() (mostly wins)
    """

    p1_score = 0.0
    p2_score = 0.0

    invert_mask = sys.maxsize
    p1_tokens   = bitboard[0] & (bitboard[1] ^ invert_mask)
    p2_tokens   = bitboard[0] & (bitboard[1])

    for n in range(1):  # allow short circuit break statement
        p1_bitmasks = p1_tokens & gameovers
        p2_bitmasks = p2_tokens & gameovers
        p1_wins     = p1_bitmasks == gameovers
        p2_wins     = p2_bitmasks == gameovers

        # If we have 4 in a row, then immediately return infinity
        if np.any(p1_wins):
            p1_score = np.inf
            break
        if np.any(p2_wins):
            p2_score = np.inf
            break

        # Exclude any gameovers that contain moves from both players
        # np.log2() % 1 == 0 will be true for any bitmask containing only a single bit
        # p1_lines is a shorted array only containing matching gameover masks
        overlaps            = (p1_bitmasks != 0) & (p2_bitmasks != 0)
        p1_is_valid         = (p1_bitmasks != 0) & ~overlaps
        p2_is_valid         = (p2_bitmasks != 0) & ~overlaps
        p1_is_single_square = p1_is_valid & ( np.log2(p1_bitmasks) % 1 == 0 )
        p2_is_single_square = p2_is_valid & ( np.log2(p2_bitmasks) % 1 == 0 )

        p1_lines            = gameovers[ p1_is_valid & ~p1_is_single_square ]
        p2_lines            = gameovers[ p2_is_valid & ~p2_is_single_square ]

        # Offer a reduced score for any bitmask containing only a single bit
        p1_score += p1_lines.size + np.count_nonzero(p1_is_single_square) * single_square_score
        p2_score += p2_lines.size + np.count_nonzero(p2_is_single_square) * single_square_score

        # NOTE: Turns out that trying to be clever and creating a square matrix using np.roll() is actually slower!
        if p1_lines.size >= 2:
            for n1 in range(p1_lines.size):
                for n2 in range(n1+1, p1_lines.size):
                    gameover1 = p1_lines[n1]
                    gameover2 = p1_lines[n2]
                    overlap   = gameover1 & gameover2
                    if overlap == 0:              continue  # Ignore no overlap
                    if np.log2(overlap) % 1 != 0: continue  # Only count double_attacks with a single overlap square
                    p1_score += double_attack_score
        if p2_lines.size >= 2:
            for n1 in range(p2_lines.size):
                for n2 in range(n1+1, p2_lines.size):
                    gameover1 = p2_lines[n1]
                    gameover2 = p2_lines[n2]
                    overlap   = gameover1 & gameover2
                    if overlap == 0:              continue  # Ignore no overlap
                    if np.log2(overlap) % 1 != 0: continue  # Only count double_attacks with a single overlap square
                    p2_score += double_attack_score

    score = (p1_score - p2_score) if player_id == 1 else (p2_score - p1_score)
    # assert np.math.isclose( score, bitboard_gameovers_heuristic_unvectorized(bitboard, player_id, gameovers), abs_tol=0.01), f'{score} != {bitboard_gameovers_heuristic_unvectorized(bitboard, player_id, gameovers)}'

    # global min_score, max_score
    # if score < min_score: min_score = score; print(f'min_score: {min_score}')  # min_score: -32.3
    # if score > max_score: max_score = score; print(f'max_score: {max_score}')  # max_score:  26.4
    return score



# @njit
def bitboard_gameovers_heuristic_unvectorized( bitboard: np.ndarray, player_id: int, gameovers: np.ndarray = get_gameovers() ) -> float:
    invert_mask = sys.maxsize
    p1_tokens   = bitboard[0] & (bitboard[1] ^ invert_mask)
    p2_tokens   = bitboard[0] & (bitboard[1])

    p1_score = 0.0
    p2_score = 0.0
    p1_gameovers = []
    p2_gameovers = []
    for gameover in gameovers:
        p1_can_play = p1_tokens & gameover
        p2_can_play = p2_tokens & gameover
        if p1_can_play and not p2_can_play:
            if   p1_can_play == gameover:       p1_score += np.inf                # Connect 4
            elif np.log2(p1_can_play) % 1 == 0: p1_score += single_square_score   # Mostly ignore single square lines
            else:                               p1_score += 1; p1_gameovers.append(gameover);
        elif p2_can_play and not p1_can_play:
            if   p2_can_play == gameover:       p2_score += np.inf                # Connect 4
            elif np.log2(p2_can_play) % 1 == 0: p2_score += single_square_score   # Mostly ignore single square lines
            else:                               p2_score += 1; p2_gameovers.append(gameover);


    for n1 in range(len(p1_gameovers)):
        for n2 in range(n1+1, len(p1_gameovers)):
            gameover1 = p1_gameovers[n1]
            gameover2 = p1_gameovers[n2]
            overlap = gameover1 & gameover2
            if gameover1 == gameover2:    continue  # Ignore self
            if overlap == 0:              continue  # Ignore no overlap
            if np.log2(overlap) % 1 != 0: continue  # Only count double_attacks with a single overlap square
            p1_score += double_attack_score
    for n1 in range(len(p2_gameovers)):
        for n2 in range(n1+1, len(p2_gameovers)):
            gameover1 = p2_gameovers[n1]
            gameover2 = p2_gameovers[n2]
            overlap = gameover1 & gameover2
            if gameover1 == gameover2:    continue  # Ignore self
            if overlap == 0:              continue  # Ignore no overlap
            if np.log2(overlap) % 1 != 0: continue  # Only count double_attacks with a single overlap square
            p2_score += double_attack_score

    if player_id == 1:
        return p1_score - p2_score
    else:
        return p2_score - p1_score



# Profiler (Macbook Pro 2011): This vectorized implementation is actually slower than unvectorized
# bitboard_gameovers_heuristic	                call_count=9469	    time=1558	own_time=1141
# bitboard_gameovers_heuristic_unvectorized	    call_count=9469	    time=1879	own_time=1862  ( 20% slower)
# bitboard_gameovers_heuristic_slow	            call_count=9469	    time=3419	own_time=1893  (220% slower)
def bitboard_gameovers_heuristic_slow( bitboard: np.ndarray, player_id: int, gameovers: np.ndarray = get_gameovers() ) -> float:

    # Hyperparameters
    single_square_score = 0.1  # Mostly ignore single squares, that can make lines in 8 directions
    double_attack_score = 0.5  # 0.5 == 100% winrate vs AlphaBetaAgent

    scores = np.array([0.0, 0.0], dtype=np.float32)

    invert_mask = sys.maxsize
    tokens = np.array([
        bitboard[0] & (bitboard[1] ^ invert_mask),
        bitboard[0] & (bitboard[1]),
    ])
    for n in range(1):  # allow short circuit break statement
        bitmasks = np.array([ token & gameovers for token in tokens ])
        wins     = bitmasks == gameovers
        if np.any(wins):
            if np.any(wins[0]): scores[0] = np.inf
            if np.any(wins[1]): scores[1] = np.inf
            break

        is_overlap        = (bitmasks[0] != 0) & (bitmasks[1] != 0)
        is_valid          = (bitmasks    != 0) & ~is_overlap
        if not np.any(is_valid):
            break

        is_single_square = is_valid & ( np.log2(bitmasks, where=is_valid ) % 1 == 0 )
        is_multi_line    = is_valid & ~is_single_square
        scores += np.count_nonzero(is_multi_line, axis=-1) + np.count_nonzero(is_single_square, axis=-1) * single_square_score
        pass

        lines = [
            gameovers[ is_multi_line[0] ],
            gameovers[ is_multi_line[1] ]
        ]
        for player in range(len(lines)):
            player_lines = lines[player]
            if len(player_lines) <= 1: continue
            for n in range(player_lines.size):
                gameover = player_lines[n]
                overlaps = gameover & player_lines[n+1:]
                overlaps = overlaps[ (overlaps != 0) & (overlaps != gameover) ]  # Ignore empty and self
                double_attacks = np.count_nonzero(np.log2(overlaps) % 1 == 0)    # Only count double_attacks with a single overlap square
                scores[player] += double_attacks * double_attack_score

    score = (scores[0] - scores[1]) if player_id == 1 else (scores[1] - scores[0])
    # assert np.math.isclose( score, bitboard_gameovers_heuristic_unvectorized(bitboard, player_id, gameovers), abs_tol=0.01), f'{score} != {bitboard_gameovers_heuristic_unvectorized(bitboard, player_id, gameovers)}'
    return score

