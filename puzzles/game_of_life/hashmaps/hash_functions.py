import numpy as np
from numba import njit

from hashmaps.primes import hashable_primes
from hashmaps.primes import primes_np


@njit()
def hash_geometric(board: np.ndarray) -> int:
    """
    Takes the 1D pixelwise view from each pixel (up, down, left, right) with wraparound
    the distance to each pixel is encoded as a prime number, the sum of these is the hash for each view direction
    the hash for each cell is the product of view directions and the hash of the board is the sum of these products
    this produces a geometric invariant hash that will be identical for roll / flip / rotate operations
    """
    assert board.shape[0] == board.shape[1]  # assumes square board
    size     = board.shape[0]
    l_primes = hashable_primes[:size//2+1]   # each distance index is represented by a different prime
    r_primes = l_primes[::-1]                # symmetric distance values in reversed direction from center

    hashed = 0
    for x in range(size):
        for y in range(size):
            # current pixel is moved to center [13] index
            horizontal = np.roll( board[:,y], size//2 - x)
            vertical   = np.roll( board[x,:], size//2 - y)
            left       = np.sum( horizontal[size//2:]   * l_primes )
            right      = np.sum( horizontal[:size//2+1] * r_primes )
            down       = np.sum( vertical[size//2:]     * l_primes )
            up         = np.sum( vertical[:size//2+1]   * r_primes )
            hashed    += left * right * down * up
    return hashed


@njit()
def hash_translations(board: np.ndarray) -> int:
    """
    Takes the 1D pixelwise view from each pixel (left, down) with wraparound
    by only using two directions, this hash is only invariant for roll operations, but not flip or rotate
    this allows determining which operations are required to solve a transform

    NOTE: np.rot180() produces the same sum as board, but with different numbers which is fixed via: sorted * primes
    """
    assert board.shape[0] == board.shape[1]
    hashes = hash_translations_board(board)
    sorted = np.sort(hashes.flatten())
    hashed = np.sum(sorted[::-1] * primes_np[:len(sorted)])  # multiply big with small numbers | hashable_primes is too small
    return int(hashed)


@njit()
def hash_translations_board(board: np.ndarray) -> np.ndarray:
    """ Returns a board with hash values for individual cells """
    assert board.shape[0] == board.shape[1]  # assumes square board
    size = board.shape[0]

    # NOTE: using the same list of primes for each direction, results in the following identity splits:
    # NOTE: np.rot180() produces the same np.sum() hash, but using different numbers which is fixed via: sorted * primes
    #   with v_primes == h_primes and NOT sorted * primes:
    #       identity == np.roll(axis=0) == np.roll(axis=1) == np.rot180()
    #       np.flip(axis=0) == np.flip(axis=1) == np.rot90() == np.rot270() != np.rot180()
    #   with v_primes == h_primes and sorted * primes:
    #       identity == np.roll(axis=0) == np.roll(axis=1)
    #       np.flip(axis=0) == np.rot270()
    #       np.flip(axis=1) == np.rot90()
    h_primes = hashable_primes[ 0*size : 1*size ]
    v_primes = hashable_primes[ 1*size : 2*size ]
    output   = np.zeros(board.shape, dtype=np.int64)
    for x in range(size):
        for y in range(size):
            # current pixel is moved to left [0] index
            horizontal  = np.roll( board[:,y], -x )
            vertical    = np.roll( board[x,:], -y )
            left        = np.sum( horizontal * h_primes )
            down        = np.sum( vertical   * v_primes )
            output[x,y] = left * down
    return output


@njit
def max_index_x(board: np.array): return max_index(board, axis=0)
@njit
def max_index_y(board: np.array): return max_index(board, axis=1)
@njit
def max_index(board: np.array, axis=0):
    """
    Find the index column with with the most number of cells, but with a unique cell count
    This should hopefully give us a reference location for translations between geometric hashes
    NOTE: this won't always work, such as for a 2x2 square, which could also be in the corner point
    """
    if axis == 0: col_counts = np.array([ np.sum(board[i,:]) for i in range(board.shape[0]) ])
    else:         col_counts = np.array([ np.sum(board[:,i]) for i in range(board.shape[1]) ])
    col_sorted = np.sort(col_counts)[::-1]
    for count in col_sorted:
        if len(col_counts[ col_counts == count ]) == 1:
            col_value = count
            break
    else:
        col_value = col_sorted[0]  # if we can't find a unique value, then just return the maximum value
    col_index = np.where( col_counts == col_value )[0][0]
    return col_index
