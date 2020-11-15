import matplotlib.pyplot as plt
import numpy as np

from utils.util import csv_to_delta, csv_to_numpy


def plot_3d(solution_3d: np.ndarray, size=4, max_cols=6, title=''):
    cols = np.min([ len(solution_3d), max_cols ])
    rows = len(solution_3d) // cols + 1
    plt.figure(figsize=(cols*size, rows*size))
    plt.title(title)
    for t in range(len(solution_3d)):
        board = solution_3d[t]
        plt.subplot(rows, cols, t + 1)
        plt.imshow(board, cmap='binary'); plt.title(f't={t}')
    plt.show()


def plot_idx(df, idx: int, size=4):
    # pd.read_csv(index_col='id') implies offset of 1 (original code uses offset = 2)
    delta = csv_to_delta(df, idx)
    start = csv_to_numpy(df, idx, key='start')
    stop  = csv_to_numpy(df, idx, key='stop')
    if len(start) == 0: start = np.zeros(stop.shape)
    if len(stop)  == 0: start = np.zeros(start.shape)

    plt.figure(figsize=(size*2,size))
    plt.subplot(121)
    plt.imshow(start, cmap='binary'); plt.title(f'{idx}: start = T=0')
    plt.subplot(122)
    plt.imshow(stop,  cmap='binary'); plt.title(f'{idx}: stop = T={delta}')
    plt.show()
