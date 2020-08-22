import pytest

from core.ConnectXBBNN import *
from heuristics.BitsquaresHeuristic import bitsquares_heuristic


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
    bitboard  = empty_bitboard()
    player_id = 1
    for action in actions:
        if isinstance(action, tuple): action, player_id = action
        bitboard  = result_action(bitboard, action, player_id=player_id % 2)
        player_id = next_player_id(player_id)

    p1_score  = heuristic(bitboard, player_id=1)
    p2_score  = heuristic(bitboard, player_id=2)
    assert np.math.isclose(p1_score, expected,  abs_tol=1e-8), f"{p1_score} != {expected}"
    assert np.math.isclose(p1_score, -p2_score, abs_tol=1e-8), f"{p2_score} != {expected}"

