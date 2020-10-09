# Source: https://www.kaggle.com/jamesmcguigan/game-of-life-image-segmentation-solver
import os
import pickle
import time

import humanize
import numpy as np

from image_segmentation.history_lookup import get_cluster_history_lookup
from utils.datasets import output_directory
from utils.datasets import test_df
from utils.datasets import train_df
from utils.game import generate_random_boards
from utils.util import csv_to_numpy_list

cluster_history_lookup_cachefile = f'{output_directory}/cluster_history_lookup.pickle'
try:
    if not os.path.exists(cluster_history_lookup_cachefile): raise FileNotFoundError
    with open(cluster_history_lookup_cachefile, 'rb') as file:
        cluster_history_lookup = pickle.load( file )
except:
    cluster_history_lookup = None


if __name__ == '__main__':
    time_start = time.perf_counter()

    dataset = np.concatenate([
        csv_to_numpy_list(train_df, key='start'),
        csv_to_numpy_list(test_df,  key='stop'),
        generate_random_boards(100_000)
    ])
    cluster_history_lookup = get_cluster_history_lookup(dataset, forward_play=10)

    time_taken = time.perf_counter() - time_start
    with open(cluster_history_lookup_cachefile, 'wb') as file:
        pickle.dump( cluster_history_lookup, file )
        print(f'wrote: {cluster_history_lookup_cachefile} = {humanize.naturalsize(os.path.getsize(cluster_history_lookup_cachefile))}')
    print(f'{len(cluster_history_lookup)} unique clusters in {time_taken:.1f}s = {1000*time_taken/len(dataset):.0f}ms/board')

