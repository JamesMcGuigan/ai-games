#!/usr/bin/env python3
# This runtime generation code was causing memory leaks in python,
# so keep in a separate file to avoid accidental imports

import os
import pickle

from hashmaps.reverse_patterns import generate_reverse_pattern_counts
from hashmaps.reverse_patterns import generate_reverse_pattern_lookup
from utils.datasets import output_directory

reverse_patterns_cachefile = f'{output_directory}/reverse_patterns.pickle'
try:
    if not os.path.exists(reverse_patterns_cachefile): raise FileNotFoundError
    with open(reverse_patterns_cachefile, 'rb') as file:
        (reverse_pattern_counts, reverse_pattern_lookup) = pickle.load( file )
except (FileNotFoundError, EOFError) as exception:
    print(f'{reverse_patterns_cachefile} not found: regenerating')
    reverse_pattern_counts = generate_reverse_pattern_counts()
    reverse_pattern_lookup = generate_reverse_pattern_lookup(reverse_pattern_counts)
    with open(reverse_patterns_cachefile, 'wb') as file:
        pickle.dump( (reverse_pattern_counts, reverse_pattern_lookup), file )
