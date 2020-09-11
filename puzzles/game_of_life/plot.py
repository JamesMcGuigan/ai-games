# Source: https://www.kaggle.com/ianmoone0617/reversing-conways-game-of-life-tutorial
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from matplotlib import animation
from matplotlib import rc


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


def plot_xy(X,Y, title1 = 'X',title2 = 'Y'):
    shape = (25,25)
    plt.figure(figsize=(16,6))
    plt.subplot(121)
    plt.imshow(X.reshape(shape),cmap='binary'); plt.title(title1)
    plt.subplot(122)
    plt.imshow(Y.reshape(shape),cmap='binary'); plt.title(title2)
    plt.show()


def plot_animate_fig(X, Y, delta, cmap='binary'):
    shape = (25,25)
    # plt.figure(figsize=(14,6))
    # plt.subplot(121)
    # plt.imshow(X.reshape(shape),cmap=cmap);plt.title('start stage')
    # plt.subplot(122)
    # plt.imshow(Y.reshape(shape),cmap=cmap);plt.title('stop stage')
    # plt.show()
    # print('Approximated Animation till stop stage : ')

    X = X.reshape(shape)
    Y = Y.reshape(shape)
    X_blank = np.zeros_like(X)

    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure(figsize=(38,38), dpi=10)
    ax  = fig.add_axes([0, 0, 1, 1], xticks=[], yticks=[], frameon=False)
    im  = ax.imshow(X,cmap=cmap)
    im.set_clim(-0.05, 1)

    def init():
        im.set_data(X_blank)
        return (im,)

    # animation function.  This is called sequentially
    def animate(i):
        im.set_data(animate.X)
        animate.X = life_step(animate.X)
        return (im,)

    animate.X = X

    ### call the animator. blit=True means only re-draw the parts that have changed.

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=delta, interval=300, blit=True)

    # equivalent to rcParams['animation.html'] = 'html5'
    rc('animation', html='jshtml')
    return HTML(anim.to_jshtml())


def life_step_1(X):
    """Game of life step using generator expressions"""
    nbrs_count = sum(np.roll(np.roll(X, i, 0), j, 1)
                     for i in (-1, 0, 1) for j in (-1, 0, 1)
                     if (i != 0 or j != 0))
    return (nbrs_count == 3) | (X & (nbrs_count == 2))


def life_step_2(X):
    """Game of life step using scipy tools"""
    from scipy.signal import convolve2d
    nbrs_count = convolve2d(X, np.ones((3, 3)), mode='same', boundary='wrap') - X
    return (nbrs_count == 3) | (X & (nbrs_count == 2))

life_step = life_step_2