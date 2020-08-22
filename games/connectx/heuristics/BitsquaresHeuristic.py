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
        player_tokens  = [ bitboard[0] & ~bitboard[1], bitboard[0] &  bitboard[1] ]
        playable_lines = get_playable_lines(bitboard)
        played_bits    = [ player_tokens[0] & playable_lines[0], player_tokens[1] & playable_lines[1] ]

        scores = [ 0, 0 ]
        for player in [0,1]:
            is_gameover  = (playable_lines[player][:] == played_bits[player][:])
            is_singlebit = np.log2(played_bits[player][:]) % 1 == 0
            to_count     = played_bits[player][ ~is_gameover & ~is_singlebit ]
            bitcounts    = np.array([ np.count_nonzero(bitcount_mask[:] & bitmask) for bitmask in to_count[:] ])
            bitsquares   = bitcounts ** reward_power

            scores[player] += np.inf if np.count_nonzero(is_gameover[:]) else 0
            scores[player] += 1 * np.count_nonzero(is_singlebit[:])
            scores[player] += np.sum( bitsquares[:] ) if len(bitsquares) else 0

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
