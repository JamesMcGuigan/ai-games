import numpy as np


def color_classes(a):
    b = len(np.nonzero(np.unique(a))[0])
    return b


# color classes (not include 0)
def take_negative(a0):
    a = np.array(a0)
    a_copy = a.copy()
    if color_classes(a) == 2:
        if 0 in a:
            c1, c2 = np.unique(a)[1], np.unique(a)[2]
        else:
            c1, c2 = np.unique(a)[0], np.unique(a)[1]

        for i in range(len(a0)):
            for j in range(len(a0[0])):
                if a[i][j] == c1:
                    a_copy[i][j] = c2
                elif a[i][j] == c2:
                    a_copy[i][j] = c1

        return a_copy.tolist()

    else:
        return -1
