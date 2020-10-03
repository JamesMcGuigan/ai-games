import numpy as np
import pytest

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
]

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
    inputs   = generate_random_boards(10_000)
    expected = life_steps(inputs)
    output   = model.predict(inputs)
    assert np.array_equal( output, expected )  # assert 100% accuracy
