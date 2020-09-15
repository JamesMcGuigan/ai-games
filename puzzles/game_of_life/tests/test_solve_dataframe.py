# import numpy as np
# from joblib import delayed
# from joblib import Parallel
#
# from constraint_satisfaction.solve_dataframe import solve_board_idx
# from utils.datasets import train_df
# from utils.hashmaps import hash_translations
# from utils.idx_lookup import get_invarient_idxs
# from utils.util import csv_to_delta
# from utils.util import csv_to_numpy
#
#
# def test_solve_dataframe():
#     df       = train_df
#     all_idxs = get_invarient_idxs(train_df)  # len(invarient_idxs) == 2961
#     print('len(invarient_idxs)', len(all_idxs), all_idxs)
#     hashes   = [ hash_translations(csv_to_numpy(df, idx, key='stop')) for idx in all_idxs ]
#     dataset  = { hashed: idx for idx, hashed in zip(all_idxs, hashes) }
#
#     idxs     = list(dataset.values())[:10]   # len(invarient_idxs) == 819
#     print('len(idxs)', len(idxs), idxs)
#     deltas   = [ csv_to_delta(df, idx)              for idx in idxs ]
#     starts   = [ csv_to_numpy(df, idx, key='start') for idx in idxs ]
#     stops    = [ csv_to_numpy(df, idx, key='stop')  for idx in idxs ]
#     boards__idxs = Parallel(-1)([
#         delayed(solve_board_idx)(board, delta, idx)
#         for board, delta, idx in zip(stops, deltas, idxs)
#     ])
#     for n, (board, idx) in enumerate(boards__idxs):
#         assert np.all( board == starts[n] )
#
# if __name__ == '__main__':
#     test_solve_dataframe()
