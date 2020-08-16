from core.ConnectXBBNN import *

### Predefine masks

# 7x1 Bitmasks representing the individual columns
mask_columns = np.array([
    np.sum([ 1 << (col + row * configuration.columns) for row in range(configuration.rows) ])
    for col in range(configuration.columns)
], dtype=np.int64)

def get_oddeven_bitboard( bitboard: np.ndarray ) -> np.ndarray:
    # This is the slow but simple method, loop over the grid and draw the odd/even bit pixels one by one
    bitcount_mask = get_bitcount_mask(configuration.columns * configuration.rows)
    column_values = (bitboard[0] & mask_columns[:]) >> np.arange(0,len(mask_columns))
    column_counts = np.array([ np.count_nonzero( bitcount_mask[:] & value ) if value else 0 for value in column_values ])
    column_spaces = configuration.rows - column_counts
    oddeven       = np.array([0,0], dtype=np.int64)
    for col in range(configuration.columns):
        for height, row in enumerate(range(column_spaces[col]-1, -1, -1)):
            index = (col + row * configuration.columns)
            if height % 2 == 1:  # is odd
                oddeven[0] |= (1 << index)  # odd
            else:
                oddeven[1] |= (1 << index)  # even
    return oddeven


# # 7x3 bitmasks indicating the odd pixels in each column
# mask_is_odd = np.array([
#     [
#         1 << (col + row * configuration.columns)
#         for row in range(configuration.rows)
#         if row % 2 == 0
#     ]
#     for col in range(configuration.columns)
# ], dtype=np.int64)
#
# # 7x3 bitmasks indicating the even pixels in each column
# mask_is_even = np.array([
#     [
#         1 << (col + row * configuration.columns + 1)
#         for row in range(configuration.rows)
#         if row % 2 == 0
#     ]
#     for col in range(configuration.columns)
# ], dtype=np.int64)
#
# # 2x1 Oddeven bitmasks for the first column
# mask_oddeven = np.array([
#     np.sum([ 1 << (row * configuration.columns) for row in range(1, configuration.rows, 2) ]),  # Even
#     np.sum([ 1 << (row * configuration.columns) for row in range(0, configuration.rows, 2) ]),  # Odd
# ], dtype=np.int64)
#
# # 7x2 Oddeven bitmasks[col, oddeven] aligned with columns
# mask_oddeven_columns = np.array([
#     mask_oddeven[:] << (col * configuration.rows)
#     for col in range(configuration.columns)
# ])


### This version is buggy
# def get_oddeven_bitboard( bitboard: np.ndarray ) -> np.ndarray:
#     """
#     Create bitmasks for odd/even pixels relative to the current floor.
#     This will help calculate if a move can be forced.
#     """
#     columns_played = bitboard[0] & mask_columns[:]   # extract out individual columns
#
#     # If there are more played odd rows than even rows, then the column is odd=True else even=False
#     columns_is_odd = (
#           np.count_nonzero( columns_played & mask_is_odd.T,  axis=0)
#         > np.count_nonzero( columns_played & mask_is_even.T, axis=0)
#     ).astype(np.int8)
#
#     # Pick the appropriate mask from mask_oddeven_columns, given that we now know
#     odd_overlays = np.array([
#         mask_oddeven_columns[col, oddeven]
#         for col, oddeven in enumerate(columns_is_odd)
#     ], dtype=np.int64)
#     even_overlays = np.array([
#         mask_oddeven_columns[col, oddeven]
#         for col, oddeven in enumerate(~columns_is_odd)
#     ], dtype=np.int64)
#
#     # convert back to a single number
#     odd_overlay  = np.sum(odd_overlays)
#     even_overlay = np.sum(even_overlays)
#
#     # Remove played squares from output bitmasks and return as 2x1 bitboard
#     output = np.array([
#         odd_overlay  & ~bitboard[0],
#         even_overlay & ~bitboard[0],
#     ])
#     return output






def get_endgame_player_columns(bitboard: np.ndarray) -> np.ndarray:
    """
    If all other squares outside the column where played, which player would be forced to play first in this column
    Returns 7x1 int8 array, indicating player number

    TODO: test if output is right way round for p1 = 0 | p2 = 1
    QUESTION: do we need access to player_id ???
    """

    # Create a bitmask for each column, representing empty squares outside of column
    outside_of_columns_spaces = np.array([
        np.sum( mask_board & ~bitboard[0] & ~np.delete(mask_columns[:], col) )
        for col in range(configuration.columns)
    ])
    mask_bitcount = get_bitcount_mask()
    total_spaces  = np.count_nonzero( bitboard[0] & mask_bitcount[:] )
    column_spaces = np.array([
        np.count_nonzero( outside_of_columns_spaces[col] & mask_bitcount[:] )
        for col in range(configuration.columns)
    ])
    output = ((total_spaces - column_spaces) % 2 == 0).astype(np.int8)
    return output



def bitboard_oddeven_heuristic( bitboard: np.ndarray, player_id: int ) -> float:
    """
    Returns a heuristic score that attempts to accuratly predict double attacks and forced win lines
    TODO: write unit tests | this code has not even been run yet
    """
    oddeven        = get_oddeven_bitboard(bitboard)
    player_columns = get_endgame_player_columns(bitboard)
    move_number    = get_move_number(bitboard)
    # played = bitboard[0]
    # player = bitboard[1]
    # odd    = oddeven[0]
    # even   = oddeven[1]

    # played_gameovers = gameovers[ bitboard[0] & gameovers == gameovers ]  # filter out any gameovers not in play
    empty_squares     = mask_board  & ~bitboard[0]
    p1_tokens         = bitboard[0] & ~bitboard[1]
    p2_tokens         = bitboard[0] &  bitboard[1]
    p1_gameovers      = gameovers[:] &  p1_tokens
    p2_gameovers      = gameovers[:] &  p2_tokens
    p1_is_gameover    = gameovers[:] == p1_gameovers
    p2_is_gameover    = gameovers[:] == p2_gameovers

    p1_multibit      = gameovers[:] & (p1_tokens | empty_squares)
    p2_multibit      = gameovers[:] & (p2_tokens | empty_squares)
    p1_is_singlebit  = p1_multibit.astype(bool)  & (np.log2(p1_multibit) % 1 == 0)
    p2_is_singlebit  = p2_multibit.astype(bool)  & (np.log2(p2_multibit) % 1 == 0)
    p1_is_multibit   = p1_multibit.astype(bool)  & ~p1_is_singlebit
    p2_is_multibit   = p1_multibit.astype(bool)  & ~p2_is_singlebit

    p1_odd_gameovers  = gameovers[:] == gameovers[:] & (p1_tokens | oddeven[0])
    p1_even_gameovers = gameovers[:] == gameovers[:] & (p1_tokens | oddeven[1])
    p2_odd_gameovers  = gameovers[:] == gameovers[:] & (p2_tokens | oddeven[0])
    p2_even_gameovers = gameovers[:] == gameovers[:] & (p2_tokens | oddeven[1])
    p1_odd_count      = np.count_nonzero(p1_odd_gameovers)
    p1_even_count     = np.count_nonzero(p1_even_gameovers)
    p2_odd_count      = np.count_nonzero(p1_odd_gameovers)
    p2_even_count     = np.count_nonzero(p1_even_gameovers)
    p1_pair_count     = np.min([ p1_odd_count, p1_even_count ])
    p2_pair_count     = np.min([ p2_odd_count, p2_even_count ])

    reward_gameover        = np.inf
    reward_single_token    = 0.01
    reward_multi           = 0.1
    reward_oddeven_pair    = 1.0
    reward_odd_with_player = 1.0
    reward_odd_with_column = 1.0

    p1_score = 0.0
    p2_score = 0.0

    p1_score += reward_gameover     * np.count_nonzero(p1_is_gameover)
    p2_score += reward_gameover     * np.count_nonzero(p2_is_gameover)
    p1_score += reward_single_token * np.count_nonzero(p1_is_singlebit)
    p2_score += reward_single_token * np.count_nonzero(p2_is_singlebit)
    p1_score += reward_multi        * np.count_nonzero(p1_is_multibit)
    p2_score += reward_multi        * np.count_nonzero(p2_is_multibit)
    p1_score += reward_oddeven_pair * p1_pair_count
    p2_score += reward_oddeven_pair * p2_pair_count

    player_bonus = reward_odd_with_player * np.max([ p2_even_count - p2_pair_count, 0 ])
    if move_number % 2 == 0:  # player 1 to move, score from perspective of p2, evens are a win
        p2_score += player_bonus
    else:
        p1_score += player_bonus

    # TODO: figure out how to implement column bonuses
    column_bonus = 0
    p1_score += column_bonus
    p2_score += column_bonus

    score = (p1_score - p2_score) if player_id == 1 else (p2_score - p1_score)
    return score




