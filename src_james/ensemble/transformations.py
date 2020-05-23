import numpy as np

# Transformations
from src_james.ensemble.util import Defensive_Copy


def Vert(M):
    n = len(M)
    k = len(M[0])
    ans = np.zeros((n, k), dtype=int)
    for i in range(n):
        for j in range(k):
            ans[i][j] = 0 + M[n - 1 - i][j]
    return ans.tolist()


def Hor(M):
    n = len(M)
    k = len(M[0])
    ans = np.zeros((n, k), dtype=int)
    for i in range(n):
        for j in range(k):
            ans[i][j] = 0 + M[i][k - 1 - j]
    return ans.tolist()


def Rot1(M):
    n = len(M)
    k = len(M[0])
    ans = np.zeros((k, n), dtype=int)
    for i in range(n):
        for j in range(k):
            ans[j][i] = 0 + M[i][k - 1 - j]
    return ans.tolist()


def Rot2(M):
    n = len(M)
    k = len(M[0])
    ans = np.zeros((k, n), dtype=int)
    for i in range(n):
        for j in range(k):
            ans[j][i] = 0 + M[n - 1 - i][j]
    return ans.tolist()


Geometric = [[Hor, Hor], [Rot2], [Rot1, Rot1], [Rot1], [Vert], [Hor, Rot2], [Hor], [Vert, Rot2]]


def Apply_geometric(S, x):
    if S in Geometric:
        x1 = Defensive_Copy(x)
        for t in S:
            x1 = t(x1)
    return x1


def Cut(M, r1, r2):  # Cut a region into tiles
    List = []
    n = len(M)
    n1 = n // r1
    k = len(M[0])
    k1 = k // r2
    for i in range(r1):
        for j in range(r2):
            R = np.zeros((n1, k1), dtype=int)
            for t1 in range(n1):
                for t2 in range(k1):
                    R[t1, t2] = 0 + M[i * n1 + t1][j * k1 + t2]
            List.append(R.tolist())
    return List


def Glue(List, r1, r2):  # Combine tiles to one picture
    n1 = len(List[0])
    k1 = len(List[0][0])
    ans = np.zeros((n1 * r1, k1 * r2), dtype=int)
    counter = 0
    for i in range(r1):
        for j in range(r2):
            R = List[counter]
            counter += 1
            for t1 in range(n1):
                for t2 in range(k1):
                    ans[i * n1 + t1, j * k1 + t2] = 0 + R[t1][t2]
    return ans.tolist()
