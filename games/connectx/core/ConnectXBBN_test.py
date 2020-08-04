import pytest

from core.ConnectXBBNN import *



@pytest.fixture
def empty(): return empty_bitboard()
@pytest.fixture
def full():  return list_to_bitboard([1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2])  # len() == 42


def test_next_player_id():
    assert next_player_id(1) == 2
    assert next_player_id(2) == 1


def test_mirror_bitstring(empty, full):
    # Test correct mirroring of full board
    assert mirror_bitstring(empty[0]) == empty[0]
    assert mirror_bitstring(mirror_bitstring(empty[0])) == empty[0]
    assert mirror_bitstring(mirror_bitstring(full[0]))  == full[0]
    assert mirror_bitstring(mirror_bitstring(empty[1])) == empty[1]
    assert mirror_bitstring(mirror_bitstring(full[1]))  == full[1]

    # Test all combinations of starting moves for 2 players
    for action_p1_1 in range(configuration.columns):
        action_p1_2 = (configuration.columns-1) - action_p1_1
        bitboard_1 = result_action(empty, action_p1_1, 1)
        bitboard_2 = result_action(empty, action_p1_2, 1)
        assert mirror_bitstring(bitboard_1[0]) == bitboard_2[0]
        assert mirror_bitstring(bitboard_1[1]) == bitboard_2[1]
        assert mirror_bitstring(bitboard_2[0]) == bitboard_1[0]
        assert mirror_bitstring(bitboard_2[1]) == bitboard_1[1]

        for action_p2_1 in range(configuration.columns):
            action_p2_2 = (configuration.columns-1) - action_p2_1
            bitboard_3 = result_action(bitboard_1, action_p2_1, 2)
            bitboard_4 = result_action(bitboard_2, action_p2_2, 2)
            assert mirror_bitstring(bitboard_3[0]) == bitboard_4[0]
            assert mirror_bitstring(bitboard_3[1]) == bitboard_4[1]
            assert mirror_bitstring(bitboard_4[0]) == bitboard_3[0]
            assert mirror_bitstring(bitboard_4[1]) == bitboard_3[1]


def test_hash_bitboard(empty, full):
    # Test hashes for mirrored bitboard are equal

    assert hash_bitboard(empty) == (0,0)
    assert hash_bitboard(empty) != hash_bitboard(full)

    # Test all combinations of starting moves for 2 players
    for action_p1_1 in range(configuration.columns):
        action_p1_2 = (configuration.columns-1) - action_p1_1
        bitboard_1 = result_action(empty, action_p1_1, 1)
        bitboard_2 = result_action(empty, action_p1_2, 1)
        assert hash_bitboard(bitboard_1) == hash_bitboard(bitboard_2)

        for action_p2_1 in range(configuration.columns):
            action_p2_2 = (configuration.columns-1) - action_p2_1
            bitboard_3 = result_action(bitboard_1, action_p2_1, 2)
            bitboard_4 = result_action(bitboard_2, action_p2_2, 2)
            assert hash_bitboard(bitboard_3) == hash_bitboard(bitboard_4)


def test_get_move_number_and_current_player_id():
    bitboard = empty_bitboard()
    assert get_move_number(bitboard)   == 0
    assert current_player_id(bitboard) == 1

    move_number = 0
    player_id   = 1
    while not has_no_more_moves(bitboard):
        move_number += 1
        action       = np.random.choice(get_legal_moves(bitboard))
        bitboard     = result_action(bitboard, action, player_id)
        player_id    = next_player_id(player_id)
        assert get_move_number(bitboard)   == move_number
        assert current_player_id(bitboard) == player_id


def test_get_next_row():
    bitboard  = empty_bitboard()
    player_id = 1
    for action in range(configuration.columns):
        assert get_next_row(bitboard, action) == configuration.rows - 1
        for row in range(configuration.rows-1, -1, -1):
            assert get_next_row(bitboard, action) == row
            bitboard  = result_action(bitboard, action, player_id)
            player_id = next_player_id(player_id)


def test_game_result_actions():
    # Test the bottom row matches after result_action()
    player_id = 1
    bitboard  = empty_bitboard()
    bitboards = [ result_action(bitboard, action, player_id) for action in get_all_moves() ]
    results   = [ bitboard_to_numpy2d(result)[-1,:].tolist() for result in bitboards ]
    assert results == [
        [1,0,0,0,0,0,0],
        [0,1,0,0,0,0,0],
        [0,0,1,0,0,0,0],
        [0,0,0,1,0,0,0],
        [0,0,0,0,1,0,0],
        [0,0,0,0,0,1,0],
        [0,0,0,0,0,0,1],
    ]

    # Replay test, but from the perspective of the second player
    player_id = 2
    bitboard  = result_action( empty_bitboard(), action=3, player_id=1 )
    bitboards = [ result_action(bitboard, action, player_id) for action in get_all_moves() ]
    results   = [ bitboard_to_numpy2d(result)[-1,:].tolist() for result in bitboards ]
    assert results == [
        [2,0,0,1,0,0,0],
        [0,2,0,1,0,0,0],
        [0,0,2,1,0,0,0],
        [0,0,0,1,0,0,0],
        [0,0,0,1,2,0,0],
        [0,0,0,1,0,2,0],
        [0,0,0,1,0,0,2],
    ]

def test_legal_moves():
    # Fill the board up square by square to test is_legal_move() and get_legal_moves() for all possibilities
    player_id = 1
    bitboard = empty_bitboard()
    for column in range(configuration.columns):
        for row in range(configuration.rows):
            assert is_legal_move(bitboard, column), f'column={column} | row={row} | bitboard={bitboard} {bitboard_to_numpy2d(bitboard)}'
            assert get_legal_moves(bitboard).tolist() == list(range(column, configuration.columns)), f'column={column} | row={row} | bitboard={bitboard} {bitboard_to_numpy2d(bitboard)}'
            bitboard  = result_action(bitboard, column, player_id)
            player_id = next_player_id(player_id)

        assert not is_legal_move(bitboard, column), f'column={column} | bitboard={bitboard} {bitboard_to_numpy2d(bitboard)}'
        assert     get_legal_moves(bitboard).tolist() == list(range(column+1, configuration.columns)), f'column={column} | bitboard={bitboard} {bitboard_to_numpy2d(bitboard)}'
        for action in range(0, column):
            assert not is_legal_move(bitboard, action), f'action={action} | column={column} | bitboard={bitboard} {bitboard_to_numpy2d(bitboard)}'
