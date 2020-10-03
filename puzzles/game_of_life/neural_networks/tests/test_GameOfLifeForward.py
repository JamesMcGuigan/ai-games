import numpy as np

from neural_networks.hardcoded.GameOfLifeForward_128 import GameOfLifeForward_128
from utils.game import generate_random_board
from utils.game import generate_random_boards
from utils.game import life_step
from utils.game import life_steps


def test_GameOfLifeForward_single():
    model = GameOfLifeForward_128()

    # Test this works for a single board
    input    = generate_random_board()
    expected = life_step(input)
    output   = model.predict(input)
    assert np.all( output == expected )  # assert 100% accuracy



def test_GameOfLifeForward_batch(count=1000):
    model = GameOfLifeForward_128()

    # As well as in batch mode
    for _ in range(max(1,count//1000)):
        inputs   = generate_random_boards(1000)
        expected = life_steps(inputs)
        outputs  = model.predict(inputs)
        assert np.all( outputs == expected )  # assert 100% accuracy



if __name__ == '__main__':
    # GameOfLifeForward_128 can successfully predict a million boards in a row correctly
    test_GameOfLifeForward_single()
    test_GameOfLifeForward_batch(1_000_000)
    print('All tests passed!')
