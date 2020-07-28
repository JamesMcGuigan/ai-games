# CenterBot always tries to play in the center.
# This is mostly a test for what features I can get to compile inside a Kaggle Submit

# The last function defined in the file run by Kaggle in submission.csv
# BUGFIX: duplicate top-level function names in submission.py can cause a Kaggle Submission Error
import time
from struct import Struct

import numpy as np

from core.ConnectXBBNN import cast_configuration
from core.ConnectXBBNN import get_move_number
from core.ConnectXBBNN import is_legal_move
from core.ConnectXBBNN import list_to_bitboard



def CenterBot(observation: Struct, _configuration_: Struct) -> int:
    # observation   = {'mark': 1, 'board': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    # configuration = {'columns': 7, 'rows': 6, 'inarow': 4, 'steps': 1000, 'timeout': 8}

    first_move_time = 1
    safety_time     = 0.5    # Only gets checked once every hundred simulations
    start_time      = time.perf_counter()

    global configuration
    configuration = cast_configuration(_configuration_)

    player_id     = observation.mark
    listboard     = np.array( observation.board, dtype=np.int8 )  # BUGFIX: constructor fails to load data
    bitboard      = list_to_bitboard(listboard)
    move_number   = get_move_number(bitboard)
    is_first_move = int(move_number < 2)
    endtime       = start_time + _configuration_.timeout - safety_time - (first_move_time * is_first_move)

    for action in [3,2,4,1,5,0,6]:
        if is_legal_move(bitboard, action):
            time_taken = time.perf_counter() - start_time
            print(f'CenterBot() = {action} in {time_taken:0.3}s')
            return action
    return int(action)          # kaggle_environments requires a python int, not np.int32
