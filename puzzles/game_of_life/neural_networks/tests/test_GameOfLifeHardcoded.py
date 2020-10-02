import numpy as np
import pytest

from neural_networks.hardcoded.GameOfLifeHardcodedLeakyReLU import GameOfLifeHardcodedLeakyReLU
from neural_networks.hardcoded.GameOfLifeHardcodedReLU1 import GameOfLifeHardcodedReLU1

models = [
    GameOfLifeHardcodedLeakyReLU(),
    GameOfLifeHardcodedReLU1(),
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
