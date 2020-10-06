# Source: https://www.kaggle.com/jamesmcguigan/game-of-life-repeating-patterns
from collections import defaultdict
from itertools import product
from typing import Dict
from typing import List
from typing import Union

import numpy as np
from joblib import delayed
from joblib import Parallel

from hashmaps.crop import crop_and_center
from hashmaps.crop import filter_crop_and_center
from hashmaps.hash_functions import hash_geometric
from hashmaps.hash_functions import hash_translations
from utils.datasets import test_df
from utils.datasets import train_df
from utils.game import life_step
from utils.util import csv_to_numpy_list


def find_repeating_patterns(start_board: np.ndarray, delta=16, geometric=False) -> Union[np.ndarray, None]:
    """ Take 10 steps forward and check to see if the same pattern repeats """
    def hash_fn(board):
        return hash_geometric(board) if geometric else hash_translations(board)
    def is_symmetric(board):
        return np.all( board == np.flip(board, axis=0) ) or np.all( board == np.flip(board, axis=0) )

    solution_3d = [ start_board ]
    hashes      = [ hash_fn(start_board) ]
    symmetric   = is_symmetric(start_board)
    for t in range(delta):
        next_board = life_step(solution_3d[-1])
        next_hash  = hash_fn(next_board)
        symmetric  = symmetric and is_symmetric(next_board)
        solution_3d.append(next_board)
        hashes.append( hash_fn(next_board) )
        if np.count_nonzero(next_board) == 0: break  # ignore dead boards
        if next_hash in hashes[:-1]:
            return np.array(solution_3d)

    if symmetric and len(solution_3d) > 5:
        return np.array(solution_3d, dtype=np.int8)
    return None


def dataset_patterns() -> List[np.ndarray]:
    boards = np.concatenate([
        csv_to_numpy_list(train_df, key='start'),
        csv_to_numpy_list(train_df, key='stop'),
        csv_to_numpy_list(test_df,  key='stop'),
    ]).astype(np.int8)
    boards   = Parallel(-1)([ delayed(filter_crop_and_center)(board, max_size=6, shape=(25,25)) for board in boards ])
    boards   = [ board for board in boards if board is not None ]
    hashes   = Parallel(-1)([ delayed(hash_geometric)(board) for board in boards ])
    boards   = list({ hashed: board for hashed, board in zip(hashes, boards) }.values())  # deduplicate
    patterns = Parallel(-1)([ delayed(find_repeating_patterns)(board, delta=16, geometric=False) for board in boards ])
    patterns = [ pattern.astype(np.int8) for pattern in patterns if pattern is not None ]
    return patterns


def generate_boards(shape=(4,4)) -> List[np.ndarray]:
    sequences = product(range(2), repeat=np.product(shape))
    boards    = ( np.array(list(sequence), dtype=np.int8).reshape(shape) for sequence in sequences )
    boards    = ( crop_and_center(board) for board in boards )
    unique    = { board.tobytes(): board for board in boards }
    unique    = { hash_geometric(board): board for board in unique.values() }
    output    = unique.values()
    return list(output)


def generated_patterns(shape=(4,4)):
    boards   = generate_boards(shape=shape)
    boards   = list({ board.tobytes(): board for board in boards }.values())
    patterns = Parallel(-1)([ delayed(find_repeating_patterns)(board, delta=16, geometric=False) for board in boards ])
    patterns = [ pattern for pattern in patterns if pattern is not None ]
    return patterns


def grouped_patterns(patterns: List[np.ndarray]) -> Dict[bytes, np.ndarray]:
    """Group patterns by their final state"""

    # Group by 3D geometric hash
    index = {}
    for pattern_3d in patterns:
        t0_key = np.product([ hash_geometric(board) for board in pattern_3d[:] ])
        index[t0_key] = pattern_3d

    # Remove any patterns that have duplicates at T=1
    dedup = {}
    for t0_key, pattern_3d in index.items():
        t1_key = np.product([ hash_geometric(board) for board in pattern_3d[1:] ])
        if t1_key not in index:
            dedup[t0_key] = pattern_3d

    # Group by last frame
    grouped  = defaultdict(list)
    for pattern_3d in dedup.values():
        order_key = hash_geometric(pattern_3d[-1])
        grouped[order_key].append(pattern_3d)
    grouped = { **grouped }  # remove defaultdict
    return grouped

