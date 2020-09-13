import numpy as np

from util.datasets import train_df
from util.hashmaps import hash_geometric
from util.hashmaps import hash_translations
from util.util import csv_to_numpy


def test_hash_geometric():
    for idx in range(1000):
        board = csv_to_numpy(train_df, idx)
        transforms = {
            "identity": board,
            "roll_0":   np.roll(board, 1, axis=0),
            "roll_1":   np.roll(board, 1, axis=1),
            "flip_0":   np.flip(board, axis=0),
            "flip_1":   np.flip(board, axis=1),
            "rot90":    np.rot90(board, 1),
            "rot180":   np.rot90(board, 2),
            "rot270":   np.rot90(board, 3),
        }
        hashes = { f'{key:8s}': hash_geometric(value) for key, value in transforms.items()}

        # all geometric transforms should produce the same hash
        assert len(set(hashes.values())) == 1


def test_hash_translations():
    for idx in range(1000):
        board = csv_to_numpy(train_df, idx)
        if np.count_nonzero(board) < 50: continue  # skip small symmetric boards
        transforms = {
            "identity": board,
            "roll_0":   np.roll(board, 13, axis=0),
            "roll_1":   np.roll(board, 13, axis=1),
            "flip_0":   np.flip(board, axis=0),
            "flip_1":   np.flip(board, axis=1),
            "rot90":    np.rot90(board, 1),
            "rot180":   np.rot90(board, 2),
            "rot270":   np.rot90(board, 3),
        }
        hashes  = { key: hash_translations(value) for key, value in transforms.items()  }

        # rolling the board should not change the hash, but other transforms should
        assert hashes['identity'] == hashes['roll_0']
        assert hashes['identity'] == hashes['roll_1']

        # all other flip / rotate transformations should produce different hashes
        assert hashes['identity'] != hashes['flip_0']
        assert hashes['identity'] != hashes['flip_1']
        assert hashes['identity'] != hashes['rot90']
        assert hashes['identity'] != hashes['rot180']
        assert hashes['identity'] != hashes['rot270']
        assert hashes['flip_0'] != hashes['flip_1'] != hashes['rot90']  != hashes['rot180'] != hashes['rot270']

