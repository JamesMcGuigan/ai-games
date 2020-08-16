from heuristics.BitboardOddEvenHeuristic import *


def test_get_evenodd_bitboard__empty():
    """ The evenodd of an empty bitboard should be a checkerboard pattern """
    bitboard      = empty_bitboard()
    evenodd       = get_evenodd_bitboard(bitboard)
    bitcount_mask = get_bitcount_mask()
    bitcount_odd  = np.count_nonzero(bitcount_mask[:] & evenodd[0])
    bitcount_even = np.count_nonzero(bitcount_mask[:] & evenodd[1])
    print( f"({bitcount_odd}) {evenodd[0]:042b} == {evenodd[1]:042b} ({bitcount_even})" )
    assert evenodd[0]    == ~evenodd[1] & mask_board  # odd and even should be the inverse of each other for empty board
    assert bitcount_odd  == bitcount_even
    assert bitcount_odd  == (configuration.columns * configuration.rows)//2
    assert bitcount_even == (configuration.columns * configuration.rows)//2
    for row in range(configuration.rows):
        assert not( evenodd[0] & (1 << row) )  # top row should not be even
        assert      evenodd[1] & (1 << row)    # top row should be odd


        bottom_row_offset = (configuration.rows-1) * configuration.columns
        assert       evenodd[0] & (1 << (row + bottom_row_offset))    # bottom row should be even
        assert not ( evenodd[1] & (1 << (row + bottom_row_offset)) )  # bottom row should not be odd


def test_get_evenodd_bitboard__triangle():
    """ Draw a 3x3 triangle in the corner, and check it produces diagonal odd/even lines and exclude played squares """
    bitboard = empty_bitboard()
    for action in [0,0,0,1,1,2]:  # 3x3 triangle
        bitboard = result_action(bitboard, action, 1)
    evenodd = get_evenodd_bitboard(bitboard)

    assert bitboard[0] & evenodd[0] == 0  # evenodd should exclude played squares
    assert bitboard[0] & evenodd[1] == 0  # evenodd should exclude played squares

    for row in range(4):
        assert      evenodd[0] & (1 << row + row * configuration.columns)     # top down diagonals should not be even
        assert not( evenodd[1] & (1 << row + row * configuration.columns) )   # top down diagonals should be odd

    for row in range(4, configuration.columns):
        assert not( evenodd[0] & (1 << row) )  # top row (not above triangle) should not be even
        assert      evenodd[1] & (1 << row)    # top row (not above triangle) should be odd




def test_get_endgame_evenodd_columns__empty():
    bitboard       = empty_bitboard()
    player_columns = get_endgame_evenodd_columns(bitboard)
    assert player_columns.tolist() == [ 0 ] * configuration.columns


def test_get_endgame_player_columns__triangle():
    bitboard = empty_bitboard()
    for action in [0,0,0,1,1,2]:  # 3x3 triangle
        bitboard = result_action(bitboard, action, 1)

    player_columns = get_endgame_evenodd_columns(bitboard)
    assert player_columns.tolist() == [ 1, 0, 1, 0, 0, 0, 0 ]