import matplotlib.pyplot as plt
import numpy as np

def plot_3d(solution_3d: np.ndarray):
    plt.figure(figsize=(len(solution_3d)*3,3))
    for t in range(len(solution_3d)):
        X = solution_3d[t]
        plt.subplot(1, len(solution_3d), t + 1)
        plt.imshow(X, cmap='binary'); plt.title(f't={t}')
    plt.show()


def plot_fig(df, idx: int):
    # pd.read_csv(index_col='id') implies offset of 1 (original code uses offset = 2)
    X = df.loc[idx][1:625+1].values
    Y = df.loc[idx][625+1:].values
    shape = (25,25)
    plt.figure(figsize=(16,6))
    plt.subplot(121)
    plt.imshow(X.reshape(shape),cmap='binary'); plt.title('start stage')
    plt.subplot(122)
    plt.imshow(Y.reshape(shape),cmap='binary'); plt.title('stop stage')
    plt.show()
