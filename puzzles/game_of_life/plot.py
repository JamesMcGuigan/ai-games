import matplotlib.pyplot as plt
import numpy as np

def plot_3d(solution_3d: np.ndarray):
    plt.figure(figsize=(16,6))
    for t in range(len(solution_3d)):
        X = solution_3d[t]
        plt.subplot(1, len(solution_3d), t + 1)
        plt.imshow(X, cmap='binary'); plt.title(f't={t}')
    plt.show()
