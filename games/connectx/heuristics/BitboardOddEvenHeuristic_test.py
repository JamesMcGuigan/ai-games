from core.ConnectXBBNN import *
from heuristics.BitboardOddEvenHeuristic import get_oddeven_bitboard


def test_get_oddeven_bitboard__empty():
    """ The oddeven of an empty bitboard should be a checkerboard pattern """
    bitboard      = empty_bitboard()
    oddeven       = get_oddeven_bitboard(bitboard)
    bitcount_mask = get_bitcount_mask()
    bitcount_odd  = np.count_nonzero(bitcount_mask[:] & oddeven[0])
    bitcount_even = np.count_nonzero(bitcount_mask[:] & oddeven[1])
    print( f"({bitcount_odd}) {oddeven[0]:042b} == {oddeven[1]:042b} ({bitcount_even})" )
    assert oddeven[0]    == ~oddeven[1] & mask_board  # odd and even should be the inverse of each other for empty board
    assert bitcount_odd  == bitcount_even
    assert bitcount_odd  == (configuration.columns * configuration.rows)//2
    assert bitcount_even == (configuration.columns * configuration.rows)//2
    for row in range(configuration.rows):
        assert      oddeven[0] & (1 << row)    # top row should be odd
        assert not( oddeven[1] & (1 << row) )  # top row should not be even

        bottom_row_offset = (configuration.rows-1) * configuration.columns
        assert not ( oddeven[0] & (1 << (row + bottom_row_offset)) )  # bottom row should not be odd
        assert       oddeven[1] & (1 << (row + bottom_row_offset))    # bottom row should be even


def test_get_oddeven_bitboard__columns():
    """ Draw a 3x3 triangle in the corner, and check it produces diagonal odd/even lines and exclude played squares """
    bitboard = empty_bitboard()
    for action in [0,0,0,1,1,2]:  # 3x3 triangle
        bitboard = result_action(bitboard, action, 1)
    oddeven = get_oddeven_bitboard(bitboard)

    assert bitboard[0] & oddeven[0] == 0  # oddeven should exclude played squares
    assert bitboard[0] & oddeven[1] == 0  # oddeven should exclude played squares

    for row in range(4):
        assert not( oddeven[0] & (1 << row + row * configuration.columns) )   # top down diagonals should be odd
        assert      oddeven[1] & (1 << row + row * configuration.columns)     # top down diagonals should not be even

    for row in range(4, configuration.columns):
        assert      oddeven[0] & (1 << row)    # top row (not above triangle) should be odd
        assert not( oddeven[1] & (1 << row) )  # top row (not above triangle) should not be even
