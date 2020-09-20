# Reference: The First 10,000 Primes - https://primes.utm.edu/lists/small/10000.txt

import math
import sys

import numpy as np
from numba import njit


# Source: https://stackoverflow.com/questions/11619942/print-series-of-prime-numbers-in-python
def generate_primes(count):
    primes = [2]
    for n in range(3, sys.maxsize, 2):
        if len(primes) >= count: break
        if all( n % i != 0 for i in range(3, int(math.sqrt(n))+1, 2) ):
            primes.append(n)
    return primes

primes     = generate_primes(10_000)
primes_np  = np.array(primes, dtype=np.int64)
primes_set = set(primes_np)


@njit()
def generate_hashable_primes(size=50, combinations=2) -> np.ndarray:
    """
    Return a list of primes that have no summation collisions for N=2, for use in hashing
    NOTE: size > 50 or combinations > 2 produces no results
    """
    domain     = primes_np
    # domain     = list(range(100000))
    candidates = np.zeros((0,), dtype=np.int64)
    exclusions = set()
    while len(candidates) < size:  # loop until we have successfully filled the buffer
        # refill candidates ignoring exclusions
        for n in range(len(domain)):
            prime = np.int64( domain[n] )
            if not np.any( candidates == prime ):
                if prime not in exclusions:
                    candidates = np.append( candidates, prime )
                    if len(candidates) >= size: break
        else:
            return np.zeros((0,), dtype=np.int64)  # prevent infinite loop if we run out of primes

        # This implements itertools.product(*[ candidates, candidates, candidates ]
        collisions = set(candidates)
        indexes    = np.array([ 0 ] * combinations)
        while np.min(indexes) < len(candidates)-1:

            # Sum N=combinations candidate primes and check for duplicate collisions, then exclude and try again
            values = candidates[indexes]
            summed = np.sum(values)
            if summed in collisions:  # then remove the largest conflicting number from the list of candidates
                exclude    = np.max(candidates[indexes])
                candidates = candidates[ candidates != exclude ]
                exclusions.add(exclude)
                break  # pick a new set of candidates and try again
            collisions.add(summed)

            # This implements itertools.product(*[ candidates, candidates, candidates ]
            indexes[0] += 1
            for i in range(len(indexes)-1):
                while np.count_nonzero( indexes == indexes[i] ) >= 2:
                    indexes[i] += 1                    # ensure indexes are unique
                if indexes[i] >= len(candidates):      # overflow to next buffer
                    indexes[i]    = np.min(indexes)+1  # for triangular iteration
                    indexes[i+1] += 1
    return candidates

assert np.array_equal(generate_hashable_primes(size=2, combinations=2), [2, 3])
# hashable_primes = generate_hashable_primes(size=50, combinations=2)
hashable_primes = np.array([
    2,     7,    23,    47,    61,     83,    131,    163,    173,    251,
    457,   491,   683,   877,   971,   2069,   2239,   2927,   3209,   3529,
    4451,  4703,  6379,  8501,  9293,  10891,  11587,  13457,  13487,  17117,
    18869, 23531, 23899, 25673, 31387,  31469,  36251,  42853,  51797,  72797,
    76667, 83059, 87671, 95911, 99767, 100801, 100931, 100937, 100987, 100999,
], dtype=np.int64)



if __name__ == '__main__':
    for permutations in [2,3,4]:
        for size in range(25,50+1,25):
            hashable_primes = generate_hashable_primes(size=size, combinations=permutations)
            print(f'{permutations} @ {size} = ', hashable_primes)
            if len(hashable_primes) == 0: break
