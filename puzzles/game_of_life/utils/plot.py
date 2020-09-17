import matplotlib.pyplot as plt
import numpy as np

from utils.util import csv_to_delta
from utils.util import csv_to_numpy


def plot_3d(solution_3d: np.ndarray):
    plt.figure(figsize=(len(solution_3d)*3,3))
    for t in range(len(solution_3d)):
        X = solution_3d[t]
        plt.subplot(1, len(solution_3d), t + 1)
        plt.imshow(X, cmap='binary'); plt.title(f't={t}')
    plt.show()


def plot_idx(df, idx: int):
    # pd.read_csv(index_col='id') implies offset of 1 (original code uses offset = 2)
    shape = (25,25)
    delta = csv_to_delta(df, idx)
    start = csv_to_numpy(df, idx, key='start')
    stop  = csv_to_numpy(df, idx, key='stop')
    if len(start) == 0: start = np.zeros(shape)
    if len(stop)  == 0: start = np.zeros(shape)

    plt.figure(figsize=(16,6))
    plt.subplot(121)
    plt.imshow(start.reshape(shape), cmap='binary'); plt.title(f'{idx}: start = T=0')
    plt.subplot(122)
    plt.imshow(stop.reshape(shape),  cmap='binary');  plt.title(f'{idx}: stop = T={delta}')
    plt.show()
