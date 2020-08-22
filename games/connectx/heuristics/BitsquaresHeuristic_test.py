import pytest

from core.ConnectXBBNN import *
from heuristics.BitsquaresHeuristic import bitsquares_heuristic
from heuristics.BitsquaresHeuristic import get_playable_lines_by_length


def test_bitsquare_heuristic__empty():
    heuristic = bitsquares_heuristic()
    bitboard  = empty_bitboard()
    p1_score  = heuristic(bitboard, player_id=1)
    p2_score  = heuristic(bitboard, player_id=2)
    assert p1_score == 0
    assert p2_score == 0
    assert p1_score == -p2_score

@pytest.mark.parametrize("actions,expected", [
    ([],                  0),                  # empty board
    ([0],                 1*3),                # 3 single lines in corner
    ([(0,1),(0,1)],       4*1 + 1*5),          # 5 single lines (1 up, 2 across, 2 diagonal )
    ([(0,1),(0,1),(0,1)], 9*1 + 4*1 + 7*1),    # 7 single lines (1 up, 3 across, 3 diagonal )
    ([0,1,0,1,0,1],       -6*1 ),              # difference is 6 single lines on side
])
def test_bitcount_heuristic__corner(actions, expected):
    heuristic = bitsquares_heuristic(reward_power=2)  # reward_power=2 makes for simpler maths than 1.75
    bitboard  = bitboard_from_actions(actions)

    p1_score  = heuristic(bitboard, player_id=1)
    p2_score  = heuristic(bitboard, player_id=2)
    assert np.math.isclose(p1_score, expected,  abs_tol=1e-8), f"{p1_score} != {expected}"
    assert np.math.isclose(p1_score, -p2_score, abs_tol=1e-8), f"{p2_score} != {expected}"


@pytest.mark.parametrize("actions,expected", [
    ([],                  [[0,0,0,0,0], [0,0,0,0,0]]),  # empty board
    ([0],                 [[0,3,0,0,0], [0,0,0,0,0]]),  # 3 single lines in corner
    ([(0,1),(0,1)],       [[0,5,1,0,0], [0,0,0,0,0]]),  # 5 single lines (1 up, 2 across, 2 diagonal )
    ([(0,1),(0,1),(0,1)], [[0,7,1,1,0], [0,0,0,0,0]]),  # 7 single lines (1 up, 3 across, 3 diagonal )
    ([0,1,0,1,0,1],       [[0,2,1,1,0], [0,8,1,1,0]]),  # difference is 6 single lines on side
    ([0,1,0,1,0,1,0,1],   [[0,0,1,1,1], [0,9,1,1,1]]),  # 4 in a row on the side
])
def test_get_playable_lines_by_length(actions, expected):
    bitboard       = bitboard_from_actions(actions)
    playable_lines = get_playable_lines_by_length(bitboard)
    lengths = [ list(map(len, playable_lines[0])), list(map(len, playable_lines[1])) ]
    assert lengths == expected

    # Assert all lines returned have 4 bits
    for player in [0,1]:
        for n in range(len(playable_lines[player])):
            for line in playable_lines[player][n]:
                assert np.count_nonzero( bitcount_mask[:] & line ) == configuration.inarow