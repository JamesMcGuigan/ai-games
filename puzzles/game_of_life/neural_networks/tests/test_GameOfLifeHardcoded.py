import numpy as np
import pytest

from neural_networks.hardcoded.GameOfLifeForward_1 import GameOfLifeForward_1
from neural_networks.hardcoded.GameOfLifeForward_11N import GameOfLifeForward_11N
from neural_networks.hardcoded.GameOfLifeForward_128 import GameOfLifeForward_128
from neural_networks.hardcoded.GameOfLifeForward_1N import GameOfLifeForward_1N
from neural_networks.hardcoded.GameOfLifeForward_2 import GameOfLifeForward_2
from neural_networks.hardcoded.GameOfLifeForward_2N import GameOfLifeForward_2N
from neural_networks.hardcoded.GameOfLifeForward_4 import GameOfLifeForward_4
from neural_networks.hardcoded.GameOfLifeHardcodedLeakyReLU import GameOfLifeHardcodedLeakyReLU
from neural_networks.hardcoded.GameOfLifeHardcodedReLU1_21 import GameOfLifeHardcodedReLU1_21
from neural_networks.hardcoded.GameOfLifeHardcodedReLU1_41 import GameOfLifeHardcodedReLU1_41
from neural_networks.hardcoded.GameOfLifeHardcodedTanh import GameOfLifeHardcodedTanh
from utils.game import generate_random_boards
from utils.game import life_steps

models = [
    GameOfLifeHardcodedLeakyReLU(),
    GameOfLifeHardcodedReLU1_41(),
    GameOfLifeHardcodedReLU1_21(),
    GameOfLifeHardcodedTanh(),

    GameOfLifeForward_128(),
    GameOfLifeForward_4(),
    GameOfLifeForward_2(),
    GameOfLifeForward_2N(),
    GameOfLifeForward_1(),
    GameOfLifeForward_1N(),
    GameOfLifeForward_11N(),
]
boards_t0  = generate_random_boards(10_000)
boards_t1 = life_steps(boards_t0)



@pytest.mark.parametrize("boards", [
    [[ 0,0,0,0,0,
       0,0,1,0,0,
       0,1,1,1,0,
       0,0,1,0,0,
       0,0,0,0,0,],
     [ 0,0,0,0,0,
       0,1,1,1,0,
       0,1,0,1,0,
       0,1,1,1,0,
       0,0,0,0,0,]
     ],
])
@pytest.mark.parametrize("model", models)
def test_GameOfLifeHardcoded_sequences(model, boards):
    boards = np.array(boards).reshape((-1,5,5))
    for n in range(len(boards)-1):
        input    = boards[n]
        expected = boards[n+1]
        output   = model.predict(input)
        assert np.array_equal( output, expected )  # assert 100% accuracy



@pytest.mark.parametrize("board,delta", [
    ([ 0,0,0,0,0,
       0,0,0,0,0,
       0,1,1,1,0,
       0,0,0,0,0,
       0,0,0,0,0,], 2),

    ([ 0,0,0,0,0,
       0,1,1,0,0,
       0,1,1,0,0,
       0,0,0,0,0,
       0,0,0,0,0,], 1),

    ([ 0,0,0,0,0,
       0,0,0,1,0,
       0,1,0,1,0,
       0,0,1,1,0,
       0,0,0,0,0,], 20),
])
@pytest.mark.parametrize("model", models)
def test_GameOfLifeHardcoded_repeating_patterns(model, board, delta):
    board    = np.array(board).reshape(5,5)
    expected = board
    output   = board
    for n in range(delta):
        output = model.predict(output)
    assert np.array_equal( output, expected )  # assert 100% accuracy


@pytest.mark.parametrize("model", models)
def test_GameOfLifeHardcoded_generated_boards(model):
    inputs   = boards_t0
    expected = boards_t1
    outputs  = model.predict(inputs)
    assert np.array_equal( outputs, expected )  # assert 100% accuracy
