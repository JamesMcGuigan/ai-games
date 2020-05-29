import numpy as np

#array
def resize_o(a,r1,r2):
    try:
        return np.repeat(np.repeat(a, r1, axis=0), r2, axis=1)
    except:
        return a

def resize_c(a):
    c = np.count_nonzero(np.bincount(a.flatten(),minlength=10)[1:])
    return np.repeat(np.repeat(a, c, axis=0), c, axis=1)

def resize_2(a):
    return np.repeat(np.repeat(a, 2, axis=0), 2, axis=1)

def resize_3(a):
    return np.repeat(np.repeat(a, 2, axis=0), 2, axis=1)

resize_type=[resize_o,resize_c]