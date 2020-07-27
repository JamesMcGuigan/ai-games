import pytest

from agents.MontyCarlo.MontyCarloTreeSearch import new_state
from agents.MontyCarlo.MontyCarloTreeSearch import run_search
from core.ConnectXBBNN import *



@pytest.fixture
def board_p1_one_move_to_win():
    return list_to_bitboard([
        0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,
        0,0,0,1,0,0,0,
        0,0,2,1,0,0,0,
        0,0,2,1,2,0,0,
    ])
def test_p1_one_move_to_win(board_p1_one_move_to_win):
    state     = new_state()
    player_id = 1
    action, count = run_search(state, board_p1_one_move_to_win, player_id, iterations=1000)
    assert action == 3
    assert 1000 <= count <= 1100

    # action, count = run_search(state, board_p1_one_move_to_win, player_id, iterations=10000)
    # assert action == 3
    # assert 10000 <= count <= 10100


@pytest.fixture
def board_p2_one_move_to_lose():
    return list_to_bitboard(typed.List([
        0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,
        0,0,0,1,0,0,0,
        0,0,0,1,0,0,0,
        0,0,2,1,2,0,0,
    ]))
def test_p2_one_move_to_lose(board_p2_one_move_to_lose):
    state     = new_state()
    player_id = 2
    action, count = run_search(state, board_p2_one_move_to_lose, player_id, iterations=1000)
    assert action == 3
    assert 1000 <= count <= 1100

    # action, count = run_search(state, board_p2_one_move_to_lose, player_id, iterations=10000)
    # assert action == 3
    # assert 10000 <= count <= 10010






