import numpy as np

from utils.datasets import train_df
from utils.game import life_step
from utils.game import life_step_1
from utils.game import life_step_2
from utils.util import csv_to_numpy


def test_life_step():
    boards = [ csv_to_numpy(train_df, idx, key='stop') for idx in range(100) ]
    for board in boards:
        assert np.all( life_step_1(board) == life_step_2(board) )
        assert np.all( life_step(board)   == life_step_1(board) )
