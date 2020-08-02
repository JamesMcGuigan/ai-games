import sys

from core.ConnectXBBNN import *



# @njit
def bitboard_gameovers_heuristic_unvectorized( bitboard: np.ndarray, player_id: int, gameovers: np.ndarray = get_gameovers() ) -> float:
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
            if   p1_can_play == gameover:       p1_score += np.inf   # Connect 4
            elif np.log2(p1_can_play) % 1 == 0: p1_score += 0.1      # Mostly ignore single square lines
            else:                               p1_score += 1; p1_gameovers.append(gameover);
        elif p2_can_play and not p1_can_play:
            if   p2_can_play == gameover:       p2_score += np.inf   # Connect 4
            elif np.log2(p2_can_play) % 1 == 0: p2_score += 0.1      # Mostly ignore single square lines
            else:                               p2_score += 1; p2_gameovers.append(gameover);

    double_attack_score = 0.5  # 0.5 == 100% winrate vs AlphaBetaAgent
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


# Profiler: bitboard_gameovers_heuristic() is 20% quicker than bitboard_gameovers_heuristic_unvectorized()
def bitboard_gameovers_heuristic( bitboard: np.ndarray, player_id: int, gameovers: np.ndarray = get_gameovers() ) -> float:

    # Hyperparameters
    single_square_score = 0.1  # Mostly ignore single squares, that can make lines in 8 directions
    double_attack_score = 0.5  # 0.5 == 100% winrate vs AlphaBetaAgent

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
        # p1_lines
        overlaps            = (p1_bitmasks != 0) & (p2_bitmasks != 0)
        p1_is_valid         = (p1_bitmasks != 0) & ~overlaps
        p2_is_valid         = (p2_bitmasks != 0) & ~overlaps
        p1_is_single_square = p1_is_valid & ( np.log2(p1_bitmasks, where=p1_is_valid ) % 1 == 0 )
        p2_is_single_square = p2_is_valid & ( np.log2(p2_bitmasks, where=p2_is_valid ) % 1 == 0 )
        p1_lines            = gameovers[ p1_is_valid & ~p1_is_single_square ]
        p2_lines            = gameovers[ p2_is_valid & ~p2_is_single_square ]

        # Offer a reduced score for any bitmask containing only a single bit
        p1_score += p1_lines.size + np.count_nonzero(p1_is_single_square) * single_square_score
        p2_score += p2_lines.size + np.count_nonzero(p2_is_single_square) * single_square_score


        ### NOTE: Turns out that trying to be clever and using np.roll() is actually slower than a nested for loop
        ### Use np.roll() to compare all bitmasks to all other bitmasks (excluding self) in a square matrix
        ### This method will double count overlaps, so the score needs to be divided by 2
        # p1_double_attacks = np.array([ p1_lines & np.roll(p1_lines,n) for n in range(1,len(p1_lines)) ])
        # p2_double_attacks = np.array([ p2_lines & np.roll(p2_lines,n) for n in range(1,len(p2_lines)) ])
        # p1_double_attacks = p1_double_attacks[ p1_double_attacks != 0              ]  # must not have zero overlap
        # p2_double_attacks = p2_double_attacks[ p2_double_attacks != 0              ]  # must not have zero overlap
        # p1_double_attacks = p1_double_attacks[ np.log2(p1_double_attacks) % 1 == 0 ]  # must have single bit of overlap
        # p2_double_attacks = p2_double_attacks[ np.log2(p2_double_attacks) % 1 == 0 ]  # must have single bit of overlap
        # p1_score         += np.count_nonzero(p1_double_attacks) / 2 * double_attack_score
        # p2_score         += np.count_nonzero(p2_double_attacks) / 2 * double_attack_score

        for n1 in range(len(p1_lines)):
            for n2 in range(n1+1, len(p1_lines)):
                gameover1 = p1_lines[n1]
                gameover2 = p1_lines[n2]
                overlap   = gameover1 & gameover2
                if overlap == 0:              continue  # Ignore no overlap
                if np.log2(overlap) % 1 != 0: continue  # Only count double_attacks with a single overlap square
                p1_score += double_attack_score
        for n1 in range(len(p2_lines)):
            for n2 in range(n1+1, len(p2_lines)):
                gameover1 = p2_lines[n1]
                gameover2 = p2_lines[n2]
                overlap   = gameover1 & gameover2
                if overlap == 0:              continue  # Ignore no overlap
                if np.log2(overlap) % 1 != 0: continue  # Only count double_attacks with a single overlap square
                p2_score += double_attack_score

    score = (p1_score - p2_score) if player_id == 1 else (p2_score - p1_score)
    # assert np.math.isclose( score, bitboard_gameovers_heuristic_unvectorized(bitboard, player_id, gameovers), abs_tol=0.01), f'{score} != {bitboard_gameovers_heuristic_unvectorized(bitboard, player_id, gameovers)}'
    return score
