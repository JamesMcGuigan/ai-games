# Source: https://www.kaggle.com/jamesmcguigan/game-of-life-repeating-patterns
import os
import pickle
from collections import defaultdict

import numpy as np
from joblib import delayed
from joblib import Parallel

from hashmaps.crop import crop_outer_3d
from hashmaps.crop import pad_board
from hashmaps.repeating_patterns import dataset_patterns
from hashmaps.translation_solver import geometric_transforms
from utils.datasets import output_directory
from utils.tuplize import tuplize


def generate_reverse_pattern_counts(patterns=None):
    """
    Generates a reverse lookup table for known Game of Life patterns
    reverse_pattern_counts[ tuplize(current) ][ tuplize(previous) ] == count
    """
    if patterns is None:
        patterns  = dataset_patterns()                 #  853 total patterns in 1.3 minutes
        # patterns += generated_patterns(shape=(4,4))  # 5469 total patterns in 3.1 minutes = too big
    patterns_cropped = [ crop_outer_3d(solution_3d) for solution_3d in patterns ]

    reverse_pattern_counts = defaultdict(lambda: defaultdict(int))
    for solution_3d in patterns_cropped:
        boards_t0 = Parallel(-1)(
            delayed(transform)( solution_3d[t] )
            for t in reversed(range(len(solution_3d)-1))
            for transform in geometric_transforms
        )
        boards_t1 = Parallel(-1)(
            delayed(transform)( solution_3d[t+1] )
            for t in reversed(range(len(solution_3d)-1))
            for transform in geometric_transforms
        )
        for previous, current in zip(boards_t0, boards_t1):
            if not np.count_nonzero(current):  continue
            if not np.count_nonzero(previous): continue
            current  = pad_board(current,  1)
            previous = pad_board(previous, 1)
            reverse_pattern_counts[ tuplize(current) ][ tuplize(previous) ] += 1

    reverse_pattern_counts = { key: { **value } for key,value in reverse_pattern_counts.items() }  # remove defaultdict
    return reverse_pattern_counts


def generate_reverse_pattern_lookup(reverse_pattern_counts, freq=0.5):
    """
    Convert reverse_pattern_counts into a more usable format:
        reverse_pattern_lookup[ tuplize(current) ] = [ np.array(previous), np.array(previous) ]

    This also filters out any less common previous patterns
    """
    reverse_pattern_lookup = defaultdict(list)
    for current, previous_counts in reverse_pattern_counts.items():
        max_value = max(previous_counts.values())
        for previous, count in previous_counts.items():
            if count >= max_value * freq:
                reverse_pattern_lookup[current] += [ np.array(previous, dtype=np.int8) ]

    return dict(reverse_pattern_lookup)  # remove defaultdict




# Caching reduces runtime down to 10s
reverse_patterns_cachefile = f'{output_directory}/reverse_patterns.pickle'
try:
    if not os.path.exists(reverse_patterns_cachefile): raise FileNotFoundError
    with open(reverse_patterns_cachefile, 'rb') as file:
        (reverse_pattern_counts, reverse_pattern_lookup) = pickle.load( file )
except (FileNotFoundError, EOFError) as exception:
    reverse_pattern_counts = generate_reverse_pattern_counts()
    reverse_pattern_lookup = generate_reverse_pattern_lookup(reverse_pattern_counts)
    with open(reverse_patterns_cachefile, 'wb') as file:
        pickle.dump( (reverse_pattern_counts, reverse_pattern_lookup), file )
