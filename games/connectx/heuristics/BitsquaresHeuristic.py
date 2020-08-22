from core.ConnectXBBNN import *


# Scores vs bitboard_gameovers_heuristic():
#   62% winrate @ reward_power=1
#   78% winrate @ reward_power=1.25
#   81% winrate @ reward_power=1.5
#   94% winrate @ reward_power=1.75
#   48% winrate @ reward_power=2
#   64% winrate @ reward_power=2.5
#   69% winrate @ reward_power=3
#   25% winrate @ reward_power=4
def bitsquares_heuristic(reward_power=1.75):
    def _bitsquares_heuristic(bitboard: np.ndarray, player_id: int):
        lines  = get_playable_lines_by_length(bitboard)
        scores = [ 0, 0 ]
        for player in [0,1]:
            for n in range(1, configuration.inarow+1):
                scores[player] += len(lines[player][n]) * (n ** reward_power)

        score = (scores[0] - scores[1]) if player_id == 1 else (scores[1] - scores[0])
        return score
    return _bitsquares_heuristic


### Utility Functions

def get_playable_lines(bitboard: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    played_squares = bitboard[0]
    empty_squares  = mask_board  & ~bitboard[0]
    p1_tokens      = bitboard[0] & ~bitboard[1]
    p2_tokens      = bitboard[0] &  bitboard[1]

    # find gameovers masks that could be filled with empty squares
    p1_playable_lines = gameovers[:] & (p1_tokens & ~p2_tokens | empty_squares)
    p2_playable_lines = gameovers[:] & (p2_tokens & ~p1_tokens | empty_squares)
    p1_is_valid = (p1_playable_lines[:] == gameovers[:]) & (p1_playable_lines[:] & played_squares != 0)
    p2_is_valid = (p2_playable_lines[:] == gameovers[:]) & (p2_playable_lines[:] & played_squares != 0)

    p1_playable_lines = p1_playable_lines[p1_is_valid]
    p2_playable_lines = p2_playable_lines[p2_is_valid]
    return p1_playable_lines, p2_playable_lines


def get_playable_lines_by_length(bitboard: np.ndarray) -> List[List[np.ndarray]]:
    """
    Returns outputs[player][length] = gameovers[bitcount == length]
    """
    outputs        = [ [], [] ]
    player_tokens  = [ bitboard[0] & ~bitboard[1], bitboard[0] &  bitboard[1] ]
    playable_lines = get_playable_lines(bitboard)
    played_bits    = [ player_tokens[0] & playable_lines[0], player_tokens[1] & playable_lines[1] ]
    inarow         = configuration.inarow

    for player in [0,1]:
        is_gameover  = (playable_lines[player][:] == played_bits[player][:])
        is_singlebit = np.log2(played_bits[player][:]) % 1 == 0
        is_multibit  = ~is_gameover[:] & ~is_singlebit[:]
        bitcounts    = np.array([
            np.count_nonzero(
                bitcount_mask[:] & playable_lines[player][n] & player_tokens[player]
            ) if is_multibit[n]
            # else 1      if is_singlebit[n]
            # else inarow if is_gameover[n]
            else 0
            for n in range(len(playable_lines[player]))
        ])

        outputs[player]         = [ np.array([], dtype=np.int64) for _ in range(inarow + 1) ]
        outputs[player][1]      = playable_lines[player][ is_singlebit ]
        outputs[player][inarow] = playable_lines[player][ is_gameover  ]
        for n in range(2, inarow):
            outputs[player][n]  = playable_lines[player][ bitcounts == n ]
    return outputs