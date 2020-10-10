# Source: https://www.kaggle.com/jamesmcguigan/game-of-life-image-segmentation-solver
import time

import numpy as np

from image_segmentation.history_lookup import get_cluster_history_lookup
from utils.datasets import output_directory
from utils.datasets import test_df
from utils.datasets import train_df
from utils.game import generate_random_boards
from utils.gzip_pickle_file import read_gzip_pickle_file
from utils.gzip_pickle_file import save_gzip_pickle_file
from utils.util import csv_to_numpy_list


def generate_cluster_history_lookup(dataset_size=250_000, verbose=True):
    time_start = time.perf_counter()

    csv_size = len(train_df.index) + len(test_df.index)
    dataset = np.concatenate([
        csv_to_numpy_list(train_df, key='start'),
        csv_to_numpy_list(test_df,  key='stop'),
        generate_random_boards(max(1, dataset_size - csv_size))
    ])[:dataset_size]
    cluster_history_lookup = get_cluster_history_lookup(dataset, forward_play=10)

    time_taken = time.perf_counter() - time_start
    if verbose: print(f'{len(cluster_history_lookup)} unique clusters in {time_taken:.1f}s = {1000*time_taken/len(dataset):.0f}ms/board')
    return cluster_history_lookup



cluster_history_lookup_cachefile = f'{output_directory}/cluster_history_lookup.pickle'
cluster_history_lookup = read_gzip_pickle_file(cluster_history_lookup_cachefile)

if __name__ == '__main__':
    cluster_history_lookup = generate_cluster_history_lookup(dataset_size=1_000_000)
    save_gzip_pickle_file(cluster_history_lookup, cluster_history_lookup_cachefile)
