from fastcache import clru_cache

from core.ConnectXBBNN import *
from heuristics.BitsquaresHeuristic import get_playable_lines_by_length


def oddeven_heuristic(reward_3_pair=1, reward_2_endgame=1, reward_3_endgame=1):
    def _oddeven_heuristic(bitboard: np.ndarray, player_id: int) -> float:
        played_squares = bitboard[0]
        empty_squares  = mask_board  & ~bitboard[0]

        tokens = [
            bitboard[0] & ~bitboard[1],
            bitboard[0] &  bitboard[1]
        ]
        oddeven_bitboards = get_oddeven_bitboard(bitboard)
        endgame_bitboards = get_endgame_bitboard(bitboard)
        endgame_columns   = get_endgame_oddeven_columns(bitboard)
        playable_lines    = get_playable_lines_by_length(bitboard)
        current_player    = current_player_index(bitboard)

        scores = [ 0, 0 ]
        for player in [ 0, 1 ]:
            # Safeguard against future changes to board size
            n3 = configuration.inarow - 1
            n2 = configuration.inarow - 2

            # 4 in a row
            is_4_gameover   = len(playable_lines[player][configuration.inarow])

            # 2 in a row
            # For a forced double attack to work, player must force capture of one or both ends via endgame position
            is_current_player  = 0 if player == current_player else 1
            is_2_endgame       = playable_lines[player][n2] == playable_lines[player][n2] & (played_squares | endgame_bitboards[is_current_player])
            is_2_endgame_count = np.count_nonzero( is_2_endgame )

            # 3 in a row + 2 odd or 2 even squares - opponent can't block both
            is_3_odd   = playable_lines[player][n3] == (playable_lines[player][n3] & (tokens[player] | oddeven_bitboards[1]))
            is_3_even  = playable_lines[player][n3] == (playable_lines[player][n3] & (tokens[player] | oddeven_bitboards[0]))
            oddeven_pair_count = np.count_nonzero(is_3_odd) // 2 + np.count_nonzero(is_3_even) // 2

            # Check for endgame columns
            # TODO: could this be done quicker using endgame_bitboard
            is_3_endgame_count = 0
            for col in range(len(mask_columns)):
                oddeven_index       = (current_player + endgame_columns[col]) % 2 == 1
                oddeven_col_bitmask = oddeven_bitboards[ endgame_columns[col] ] & mask_columns[col]
                if not oddeven_col_bitmask: continue  # skip if no empty oddeven squares that can be played

                is_3_oddeven        = [ is_3_even, is_3_odd ][ oddeven_index ]
                oddeven_lines       = playable_lines[player][n3][ is_3_oddeven ]
                oddeven_in_col      = oddeven_lines[:] & empty_squares & oddeven_col_bitmask
                is_3_endgame_count += np.count_nonzero(oddeven_in_col)


            # Calculate scores based on rewards
            scores[player] += np.inf          if is_4_gameover else 0
            scores[player] += reward_3_pair    * oddeven_pair_count
            scores[player] += reward_2_endgame * is_2_endgame_count
            scores[player] += reward_3_endgame * is_3_endgame_count

        score = (scores[0] - scores[1]) if player_id == 1 else (scores[1] - scores[0])
        return score
    return _oddeven_heuristic


### Predefine masks

# 7x1 Bitmasks representing the individual columns
mask_columns = np.array([
    np.sum([ 1 << (col + row * configuration.columns) for row in range(configuration.rows) ])
    for col in range(configuration.columns)
], dtype=np.int64)


### Utility Functions

def get_oddeven_bitboard( bitboard: np.ndarray ) -> Tuple[int,int]:
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


def get_endgame_bitboard( bitboard: np.ndarray ) -> Tuple[int,int]:
    oddeven = get_oddeven_bitboard(bitboard)
    columns = get_endgame_oddeven_columns(bitboard)
    endgame = int(np.sum([
        mask_columns[col] & oddeven[columns[col]]
        for col in range(len(mask_columns))
    ]))
    return ( endgame, ~endgame & mask_board )  # Needs to be this way round to work with with player_index


def get_endgame_oddeven_columns(bitboard: np.ndarray) -> np.ndarray:
    """
    If all other squares outside the column where played, which player would be forced to play first in this column
    Returns 7x1 int8 array, indicating player number
    """
    column_counts   = get_column_counts(bitboard)
    total_spaces    = np.sum(column_counts)
    player_turn     = total_spaces % 2  # p1 to move if number of spaces is even
    endgame_columns = (player_turn + total_spaces - column_counts[:]) % 2
    return endgame_columns

def get_column_counts(bitboard: np.ndarray) -> np.ndarray:
    """ Return the number of empty spaces in each column """
    column_spaces = (mask_board & ~bitboard[0] & mask_columns[:])
    column_counts = np.array([
        np.count_nonzero( bitcount_mask[:] & column_bitmask )
        for column_bitmask in column_spaces
    ])
    return column_counts

