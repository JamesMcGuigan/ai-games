import itertools
import time
from collections import defaultdict
from typing import List

import numpy as np
from joblib import delayed
from joblib import Parallel

from constraint_satisfaction.fix_submission import is_valid_solution
from hashmaps.hash_functions import hash_geometric
from hashmaps.translation_solver import solve_translation
from image_segmentation.clusters import extract_clusters_from_labels
from image_segmentation.clusters import label_board
from utils.datasets import sample_submission_df
from utils.util import csv_to_delta_list
from utils.util import csv_to_numpy_list
from utils.util import numpy_to_series


def image_segmentation_dataframe_solver( df, history, submission_df=None, exact=False, blank_missing=True, verbose=True ):
    time_start = time.perf_counter()
    stats      = { "partial": 0, "exact": 0, "total": 0 }

    submission_df = submission_df if submission_df is not None else sample_submission_df.copy()
    idxs       = df.index
    deltas     = csv_to_delta_list(df)
    boards     = csv_to_numpy_list(df, key='stop')
    labeleds   = Parallel(-1)( delayed(label_board)(board)                          for board in boards )
    clustereds = Parallel(-1)( delayed(extract_clusters_from_labels)(board, labels) for board, labels in zip(boards, labeleds) )

    for idx, delta, stop_board, labels, clusters in zip(idxs, deltas, boards, labeleds, clustereds):
        # image_segmentation_solver() returns multiple solutions, if any are valid, then all returned will be valid
        start_boards = image_segmentation_solver(
            stop_board, delta, history=history, blank_missing=blank_missing,
            labels=labels, clusters=clusters
        )
        if len(start_boards) == 0: continue
        start_board = start_boards[0]   # pick first solution
        is_valid    = is_valid_solution( start_board, stop_board, delta )
        if   is_valid:                         stats['exact']   += 1
        elif np.count_nonzero( start_board ):  stats['partial'] += 1
        stats['total'] += 1

        if is_valid or not exact:
            submission_df.loc[idx] = numpy_to_series(start_board, key='start')


    time_taken = time.perf_counter() - time_start
    stats['time_seconds'] = int(time_taken)
    stats['time_hours']   = round(time_taken/60/60, 2)
    if verbose: print('image_segmentation_solver()', stats)
    return submission_df



def image_segmentation_solver(stop_board, delta, history=None, blank_missing=True, labels=None, clusters=None) -> List[np.ndarray]:
    from image_segmentation.history_lookup_cache import cluster_history_lookup  # 9Mb file, don't import unless required

    history  = history  if history  is not None else cluster_history_lookup
    labels   = labels   if labels   is not None else label_board(stop_board)
    clusters = clusters if clusters is not None else extract_clusters_from_labels(stop_board, labels)

    unique_labels = np.unique(labels)
    now_hashes    = Parallel(-1)( delayed(hash_geometric)(cluster) for cluster in clusters )
    new_clusters  = defaultdict(list)
    for label, now_cluster, now_hash in zip(unique_labels, clusters, now_hashes):
        if label == 0: continue
        if np.count_nonzero(now_cluster) == 0: continue
        if history.get(now_hash,{}).get(delta,None):
            for past_hash in history[now_hash][delta].keys():  # sorted by count
                try:
                    start_cluster = history[now_hash][delta][past_hash]['start']
                    stop_cluster  = history[now_hash][delta][past_hash]['stop']
                    transform_fn  = solve_translation(stop_cluster, now_cluster) # assert np.all( transform_fn(train_board) == test_board )
                    past_cluster  = transform_fn(start_cluster)
                    new_clusters[label] += [ past_cluster ]
                    break
                except Exception as exception:
                    pass
        if not label in new_clusters and not blank_missing:
            new_clusters[label] = now_cluster


    # return list of all valid cluster permutations
    valid_outputs   = []
    partial_outputs = []
    for cluster_combination in itertools.product( *new_clusters.values() ):
        if len(cluster_combination) == 0: continue  # return empty list if no solutions
        start_board = np.zeros( stop_board.shape, dtype=np.int8 )
        for cluster in cluster_combination:
            start_board += cluster
        if np.any( start_board >= 2 ): continue  # reject combinations with overlaps
        if is_valid_solution(start_board, stop_board, delta):  valid_outputs   += [ start_board ]
        else:                                                  partial_outputs += [ start_board ]

    return valid_outputs if len(valid_outputs) else partial_outputs
