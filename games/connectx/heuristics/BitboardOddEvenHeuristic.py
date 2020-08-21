from fastcache import clru_cache

from core.ConnectXBBNN import *

### Predefine masks

# 7x1 Bitmasks representing the individual columns
mask_columns = np.array([
    np.sum([ 1 << (col + row * configuration.columns) for row in range(configuration.rows) ])
    for col in range(configuration.columns)
], dtype=np.int64)



def get_evenodd_bitboard( bitboard: np.ndarray ) -> Tuple[int,int]:
    # This is the slow but simple method, loop over the grid and draw the odd/even bit pixels one by one
    column_values = (bitboard[0] & mask_columns[:])
    even_columns = [
        get_even_column(column_value, col)  # OPTIMIZATION: cache the inner loop
        for col, column_value in enumerate(column_values)
    ]
    even = sum(even_columns)
    odd  = ~even & ~bitboard[0] & mask_board
    return (even, odd)


@clru_cache(None)  # size = 2**configuration.rows * configuration.columns == 448
def get_even_column( column_bitmask: int, col: int ) -> int:
    column_count = np.count_nonzero( bitcount_mask[:] & column_bitmask )
    column_space = configuration.rows - column_count

    even = 0
    for height, row in enumerate(range(column_space-1, -1, -1)):
        if height % 2 == 0:       # is even
            index = (col + row * configuration.columns)
            even |= (1 << index)  # set even pixel
    return even




def get_endgame_evenodd_columns(bitboard: np.ndarray) -> np.ndarray:
    """
    If all other squares outside the column where played, which player would be forced to play first in this column
    Returns 7x1 int8 array, indicating player number
    """
    column_spaces = (mask_board & ~bitboard[0] & mask_columns[:])
    column_counts = np.array([
        np.count_nonzero( bitcount_mask[:] & column_bitmask )
        for column_bitmask in column_spaces
    ])
    total_spaces   = np.sum(column_counts)
    player_columns = (total_spaces - column_counts[:]) % 2
    return player_columns


def bitboard_evenodd_heuristic(
    reward_gameover        = np.inf,
    reward_single_token    = 0.1,
    reward_multi           = 1,
    reward_evenodd_pair    = 4,
    reward_odd_with_player = 0.5,
    reward_odd_with_column = 1,
):
    def _bitboard_evenodd_heuristic( bitboard: np.ndarray, player_id: int ) -> float:
        """
        Returns a heuristic score that attempts to accuratly predict double attacks and forced win lines
        TODO: write unit tests | this code has not even been run yet
        """
        evenodd         = get_evenodd_bitboard(bitboard)
        evenodd_columns = get_endgame_evenodd_columns(bitboard)
        move_number     = get_move_number(bitboard)
        # played = bitboard[0]
        # player = bitboard[1]
        # odd    = evenodd[0]
        # even   = evenodd[1]

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

        p1_odd_gameovers    = gameovers[:] == gameovers[:] & (p1_tokens | evenodd[0])
        p1_even_gameovers   = gameovers[:] == gameovers[:] & (p1_tokens | evenodd[1])
        p2_odd_gameovers    = gameovers[:] == gameovers[:] & (p2_tokens | evenodd[0])
        p2_even_gameovers   = gameovers[:] == gameovers[:] & (p2_tokens | evenodd[1])

        # # Count the number of instances where the empty space in a multibit gameover wins when all other columns are played
        # p1_odd_column_count = np.count_nonzero([
        #     mask_columns[col] & evenodd[evenodd_columns[col]]
        #     & p1_odd_gameovers[:] & p1_is_multibit & ~p1_tokens & empty_squares
        #     for col in len(mask_columns)
        # ])
        # p2_odd_column_count = np.count_nonzero([
        #     mask_columns[col] & evenodd[evenodd_columns[col]]
        #     & p2_odd_gameovers[:] & p2_is_multibit & ~p2_tokens & empty_squares
        #     for col in len(mask_columns)
        # ])

        p1_odd_count        = np.count_nonzero(p1_odd_gameovers)
        p1_even_count       = np.count_nonzero(p1_even_gameovers)
        p2_odd_count        = np.count_nonzero(p1_odd_gameovers)
        p2_even_count       = np.count_nonzero(p1_even_gameovers)
        p1_pair_count       = np.min([ p1_odd_count, p1_even_count ])
        p2_pair_count       = np.min([ p2_odd_count, p2_even_count ])

        # 47.0/100 =  47% winrate AlphaBetaBitboard vs AlphaBetaBitboardEvenOdd
        p1_score = 0.0
        p2_score = 0.0

        # p1_score += reward_gameover     * np.count_nonzero(p1_is_gameover)
        # p2_score += reward_gameover     * np.count_nonzero(p2_is_gameover)
        p1_score += reward_single_token * np.count_nonzero(p1_is_singlebit)
        p2_score += reward_single_token * np.count_nonzero(p2_is_singlebit)
        p1_score += reward_multi        * np.count_nonzero(p1_is_multibit)
        p2_score += reward_multi        * np.count_nonzero(p2_is_multibit)
        p1_score += reward_evenodd_pair * p1_pair_count
        p2_score += reward_evenodd_pair * p2_pair_count
        p1_score += reward_evenodd_pair * p1_pair_count
        p2_score += reward_evenodd_pair * p2_pair_count
        # p1_score += reward_odd_with_column * p1_odd_column_count
        # p2_score += reward_odd_with_column * p2_odd_column_count

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

    return _bitboard_evenodd_heuristic



