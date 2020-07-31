import sys

import numpy as np
from numba import njit

from core.ConnectXBBNN import get_gameovers


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