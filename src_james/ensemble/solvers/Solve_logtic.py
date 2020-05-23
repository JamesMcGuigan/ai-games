import numpy as np

from src_james.ensemble.colors import checkColorMap, findColorMap, applyColorMap
from src_james.ensemble.split_direction import vertical_split, horizontal_split, vertical_split0, horizontal_split0
from src_james.ensemble.transformations import Vert, Hor
from src_james.ensemble.util import Defensive_Copy


def logitic_And(a, b):
    a = np.array(a)
    b = np.array(b)
    c = np.zeros(a.shape)
    if a.shape != b.shape:
        return -1
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if a[i][j] != 0 and a[i][j] == b[i][j]:
                c[i][j] = a[i][j]
    return c


def logitic_And_01(a, b):
    a = np.array(a)
    b = np.array(b)
    a = np.where(a, 1, 0)
    b = np.where(b, 1, 0)

    c = np.zeros(a.shape)
    if a.shape != b.shape:
        return -1
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if a[i][j] != 0 and a[i][j] == b[i][j]:
                c[i][j] = a[i][j]
    return c


def logitic_Or(a, b):
    a = np.array(a)
    b = np.array(b)
    c = np.zeros(a.shape)
    if a.shape != b.shape:
        return -1
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if a[i][j] == 0 and b[i][j] != 0:
                c[i][j] = b[i][j]
            elif a[i][j] != 0 and b[i][j] == 0:
                c[i][j] = a[i][j]
            elif a[i][j] != 0 and a[i][j] == b[i][j]:
                c[i][j] = a[i][j]
    return c


def logitic_Or_01(a, b):
    a = np.array(a)
    b = np.array(b)
    a = np.where(a, 1, 0)
    b = np.where(b, 1, 0)
    c = np.zeros(a.shape)
    if a.shape != b.shape:
        return -1
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if a[i][j] == 0 and b[i][j] != 0:
                c[i][j] = b[i][j]
            elif a[i][j] != 0 and b[i][j] == 0:
                c[i][j] = a[i][j]
            elif a[i][j] != 0 and a[i][j] == b[i][j]:
                c[i][j] = a[i][j]
    return c


def logitic_Xor(a, b):
    a = np.array(a)
    b = np.array(b)
    c = np.zeros(a.shape)
    if a.shape != b.shape:
        return -1
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if a[i][j] == 0 and b[i][j] != 0:
                c[i][j] = b[i][j]
            elif a[i][j] != 0 and b[i][j] == 0:
                c[i][j] = a[i][j]
    return c


def logitic_Xor_01(a, b):
    a = np.array(a)
    b = np.array(b)
    a = np.where(a, 1, 0)
    b = np.where(b, 1, 0)
    c = np.zeros(a.shape)
    if a.shape != b.shape:
        return -1
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if a[i][j] == 0 and b[i][j] != 0:
                c[i][j] = b[i][j]
            elif a[i][j] != 0 and b[i][j] == 0:
                c[i][j] = a[i][j]
    return c


# need color change
def logitic_Nor(a, b):
    a = np.array(a)
    b = np.array(b)
    c = np.zeros(a.shape)
    if a.shape != b.shape:
        return -1
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if a[i][j] == 0 and b[i][j] == 0:
                c[i][j] = 1
    return c


logitic_map = [logitic_And, logitic_Or, logitic_Nor, logitic_Xor, logitic_And_01, logitic_Or_01, logitic_Xor_01]


def Apply_logitic(S, x, y):
    if S in logitic_map:
        x1 = Defensive_Copy(x)
        y1 = Defensive_Copy(y)
        z1 = S(x1, y1)
    return z1.tolist()


def Solve_logtic(basic_task):
    Input  = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    Test_Case = Input[-1]
    Input = Input[:-1]
    for i in range(len(logitic_map)):
        S = logitic_map[i]
        solved = True
        m = 0

        for x, y in zip(Input, Output):
            if vertical_split(x, y) or horizontal_split(x, y) or vertical_split0(x, y) or horizontal_split0(x, y):
                if horizontal_split(x, y):
                    a, b = horizontal_split(x, y)
                    b1 = Vert(b)
                    mask = 0

                elif horizontal_split0(x, y):
                    a, b = horizontal_split0(x, y)
                    b1 = Vert(b)
                    mask = 0
                elif vertical_split(x, y):
                    a, b = vertical_split(x, y)
                    b1 = Hor(b)
                    mask = 1

                else:
                    a, b = vertical_split0(x, y)
                    b1 = Hor(b)
                    mask = 1

            else:
                return -1
                # print(a,b)
            logitic_x = Apply_logitic(S, a, b)
            logitic_x1 = Apply_logitic(S, a, b1)
            # print(logitic_x1)
            # color chage

            if checkColorMap(logitic_x, y) and logitic_x != y:
                m = 1
                colormap  = findColorMap(logitic_x, y)
                logitic_x = applyColorMap(logitic_x, colormap)
            if checkColorMap(logitic_x1, y) and logitic_x1 != y:
                m = 1
                colormap = findColorMap(logitic_x1, y)
                logitic_x1 = applyColorMap(logitic_x1, colormap)

            if logitic_x != y and logitic_x1 != y:
                solved = False
                break

        if solved == True:

            if logitic_x == y:
                if mask == 0 and horizontal_split(Test_Case, y):
                    a, b = horizontal_split(Test_Case, y)
                elif mask == 0 and horizontal_split0(Test_Case, y):
                    a, b = horizontal_split0(Test_Case, y)

                elif mask == 1 and vertical_split(Test_Case, y):
                    a, b = vertical_split(Test_Case, y)

                elif mask == 1 and vertical_split0(Test_Case, y):
                    a, b = vertical_split0(Test_Case, y)
                # 187
                else:
                    return -1
                logitic_Test_Case = Apply_logitic(S, a, b)

                if m == 1:
                    logitic_Test_Case = applyColorMap(logitic_Test_Case, colormap)
                # to int
                logitic_Test_Case = [[int(s) for s in a] for a in logitic_Test_Case]

                return logitic_Test_Case
            else:
                if mask == 0 and horizontal_split(Test_Case, y):
                    a, b = horizontal_split(Test_Case, y)
                    b1 = Vert(b)
                elif mask == 0 and horizontal_split0(Test_Case, y):
                    a, b = horizontal_split0(Test_Case, y)
                    b1 = Vert(b)
                elif mask == 1 and vertical_split(Test_Case, y):
                    a, b = vertical_split(Test_Case, y)
                    b1 = Hor(b)

                elif mask == 1 and vertical_split0(Test_Case, y):
                    a, b = vertical_split0(Test_Case, y)
                    b1 = Hor(b)

                else:
                    return -1

                logitic_Test_Case = Apply_logitic(S, a, b1)

                if m == 1:
                    logitic_Test_Case = applyColorMap(logitic_Test_Case, colormap)
                # to int
                logitic_Test_Case = [[int(s) for s in a] for a in logitic_Test_Case]

                return logitic_Test_Case
    return -1
