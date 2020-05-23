import numpy as np

from src_james.ensemble.util import Defensive_Copy

BACKGROUND = 0
def connect_dot_row(a0):
    a = np.array(a0)
    a_copy = a.copy()
    m, n = a.shape
    for i in range(m):
        for j in range(n):
            if a[i][j] == BACKGROUND:
                continue
            else:
                c = a[i][j]
            if j + 1 <= n - 1:
                for k in range(j + 1, n):
                    if a[i][k] == c:
                        for p in range(j + 1, k):
                            if a_copy[i, p] == BACKGROUND:
                                a_copy[i, p] = c

            if i + 1 <= m - 1:
                for l in range(i + 1, m):
                    if a[l][j] == c:
                        a_copy[i + 1:l, j] = c
                        for q in range(i + 1, l):
                            if a_copy[q, j] == BACKGROUND:
                                a_copy[q, j] = c

    return a_copy.tolist()


def solve_connect(basic_task):
    Input  = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    Test_Case = Input[-1]
    Input = Input[:-1]

    for x, y in zip(Input, Output):
        pred_y = connect_dot_row(x)
        if pred_y != y:
            return -1
    return connect_dot_row(Test_Case)
