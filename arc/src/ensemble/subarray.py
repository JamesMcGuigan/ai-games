import numpy as np

# A0 > B0
from src.ensemble.util import Defensive_Copy


def issubarray(A0, B0):
    A = np.array(A0)
    B = np.array(B0)
    a1 = A.shape[0]
    a2 = A.shape[1]
    b1 = B.shape[0]
    b2 = B.shape[1]
    if (a1 == b1 and a2 == b2) or b1 > a1 or b2 > a2 or (b1 == 1 and b2 == 1):
        return False
    c1 = a1 - b1 + 1
    c2 = a2 - b2 + 1
    for i in range(c1):
        for j in range(c2):
            if (A[i:i + b1, j:j + b2] == B).all():
                return True
    return False


def issubarray_match(basic_task):
    Input  = [ Defensive_Copy(x) for x in basic_task[0] ]
    Output = [ Defensive_Copy(y) for y in basic_task[1] ]
    Test_Case = Input[-1]
    Input = Input[:-1]
    res = True
    for i in range(len(Input)):
        A = Input[i]
        B = Output[i]
        if issubarray(A, B) == False:
            res = False
            break
    return res


def subarray_count(A, B):
    A = np.array(A)
    B = np.array(B)
    a1 = A.shape[0]
    a2 = A.shape[1]
    b1 = B.shape[0]
    b2 = B.shape[1]
    if (a1 == b1 and a2 == b2) or b1 > a1 or b2 > a2 or (b1 == 1 and b2 == 1):
        return 0
    c1 = a1 - b1 + 1
    c2 = a2 - b2 + 1
    i = 0
    j = 0
    count = 0
    t = 0
    while i < c1:
        while j < c2:
            if (A[i:i + b1, j:j + b2] == B).all():
                count += 1
                j += b1
                t = 1
            else:
                j += 1
        j = 0
        if t == 1:
            i += b2
        else:
            i += 1
    return count


# A比B大


def color_counts(a):
    a = np.array(a)
    b = np.bincount(a.flatten(), minlength=10)
    b = b[1:]
    return b


# 去黑
def color_counts_mul(C, E):
    c1 = color_counts(C)
    e1 = color_counts(E)
    m = 0
    for i in range(len(e1)):
        if c1[i] == 0 and e1[i] == 0:
            continue
        elif c1[i] < e1[i] or (c1[i] == 0 and e1[i] != 0) or (e1[i] == 0 and c1[i] != 0):
            return 0
        else:
            if c1[i] % e1[i] != 0:
                return 0
            else:
                n = c1[i] // e1[i]
                if m != 0 and m != n:
                    return 0
                m = n
    return m
