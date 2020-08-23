# Interactive Connect 4 Board: https://www.mathsisfun.com/games/connect4.html
import pytest

from core.ConnectXBBNN import *
from heuristics.OddEvenHeuristic import get_endgame_oddeven_columns
from heuristics.OddEvenHeuristic import oddeven_heuristic


def test_oddeven_heuristic__empty():
    heuristic = oddeven_heuristic()
    bitboard  = empty_bitboard()
    p1_score  = heuristic(bitboard, player_id=1)
    p2_score  = heuristic(bitboard, player_id=2)
    assert p1_score == 0
    assert p2_score == 0
    assert p1_score == -p2_score

@pytest.mark.parametrize("listboard,expected", [
    # # reward_2_endgame - p1 to move and draw
    # ([ -0,-0,-0, 0,0,0,0,
    #    -0,-0,-0, 0,0,0,0,
    #    -0,-0,-0, 0,0,0,0,
    #    -0,-0,-0, 0,0,0,0,
    #    +2,-0,-0, 0,1,1,0,
    #    +2,-0,-0, 0,1,2,0,], 0 ),

    # reward_2_endgame - p1 to move and lose
    ([ -0,-0,-0, 0,0,0,0,
       -0,-0,-0, 0,0,0,0,
       -0,-0,-0, 0,0,0,0,
       -0,-0,-0, 0,0,0,0,
       -0,-0,+1, 0,2,2,0,
       -0,-0,+1, 0,2,1,0,], -1 ),

    # reward_2_endgame - p1 to move and draw
    ([ -0,-0,-0, 0,0,0,0,
       -0,-0,-0, 0,0,0,0,
       -0,-0,-0, 0,0,0,0,
       -0,-0,-0, 0,2,1,0,
       +2,-0,-0, 0,1,1,0,
       +2,-0,-0, 0,1,2,0,], 0 ),

    # reward_2_endgame - p2 to move and lose via double attack
    ([ -0,-0,-0, 0,0,0,0,
       -0,-0,-0, 0,0,0,0,
       -0,-0,-0, 0,0,0,0,
       -0,-0,-0, 0,2,1,0,
        0,-0,-0, 0,1,1,0,
        2,-0,-0, 0,1,2,0,], 2 ),

])
def test_oddeven_heuristic(listboard, expected):
    bitboard  = list_to_bitboard(listboard)
    heuristic = oddeven_heuristic(reward_3_pair=0, reward_2_endgame=1, reward_3_endgame=0)

    p1_score  = heuristic(bitboard, player_id=1)
    p2_score  = heuristic(bitboard, player_id=2)
    assert np.math.isclose(p1_score, expected,  abs_tol=1e-8), f"{p1_score} != {expected}"
    assert np.math.isclose(p1_score, -p2_score, abs_tol=1e-8), f"{p2_score} != {expected}"


@pytest.mark.parametrize("listboard,expected", [
    # reward_3_pair - p2 to move, cannot block both lines
    ([ 0,0,0,0,0,0,0,
       0,0,0,0,0,0,0,
       2,0,0,0,0,0,0,
       2,1,2,0,0,0,0,
       1,1,1,0,0,0,0,
       1,1,2,2,0,0,0,], 1),

    # reward_3_pair - p2 to move, can block both lines
    ([ 0,0,0,0,0,0,0,
       0,0,0,0,0,0,0,
       0,0,0,0,0,0,0,
       2,1,2,0,0,0,0,
       1,1,1,0,0,0,0,
       2,1,2,0,0,0,0,], 0),
])
def test_oddeven_heuristic__reward_3_pair(listboard, expected):
    bitboard  = list_to_bitboard(listboard)
    heuristic = oddeven_heuristic(reward_3_pair=1, reward_2_endgame=0, reward_3_endgame=0)

    p1_score  = heuristic(bitboard, player_id=1)
    p2_score  = heuristic(bitboard, player_id=2)
    assert np.math.isclose(p1_score, expected,  abs_tol=1e-8), f"{p1_score} != {expected}"
    assert np.math.isclose(p1_score, -p2_score, abs_tol=1e-8), f"{p2_score} != {expected}"


@pytest.mark.parametrize("listboard,expected", [
    # reward_3_endgame - p1 to move - p1 wins
    ([ -0,-0,-0, 2,0,2,1,
       -0,-0,-0, 1,0,2,1,
       -0,-0,-0, 1,0,1,2,
       -0,-0,-0, 2,0,2,1,
       -0,-0,-0, 2,0,2,1,
       -0,-0,-0, 1,0,1,1,], 1),

    # reward_3_endgame - p2 to move - draw
    ([ -0,-0,-0, 2,0,2,0,
       -0,-0,-0, 1,0,2,1,
       -0,-0,-0, 1,0,1,2,
       -0,-0,-0, 2,0,2,1,
       -0,-0,-0, 2,0,2,1,
       -0,-0,-0, 1,0,1,1,], 0),

])
def test_oddeven_heuristic__reward_3_endgame(listboard, expected):
    bitboard  = list_to_bitboard(listboard)
    heuristic = oddeven_heuristic(reward_3_pair=0, reward_2_endgame=0, reward_3_endgame=1)

    p1_score  = heuristic(bitboard, player_id=1)
    p2_score  = heuristic(bitboard, player_id=2)
    assert np.math.isclose(p1_score, expected,  abs_tol=1e-8), f"{p1_score} != {expected}"
    assert np.math.isclose(p1_score, -p2_score, abs_tol=1e-8), f"{p2_score} != {expected}"




@pytest.mark.parametrize("actions,expected", [
    ([],                  [0,0,0,0,0,0,0]),    # empty board  - 0 = p1 to move first
    ([0],                 [1,0,0,0,0,0,0]),    # first move   - if rest of board is filled, 1=p2 is forced to play 0
    ([0,0],               [0,0,0,0,0,0,0]),    # second move  - return to starting state
    ([0,0,0,1],           [1,1,0,0,0,0,0]),    # first two columns are odd
])
def test_get_endgame_oddeven_columns(actions, expected):
    bitboard = bitboard_from_actions(actions)
    oddeven  = get_endgame_oddeven_columns(bitboard)
    assert oddeven.tolist() == expected