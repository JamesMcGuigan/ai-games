import numpy as np

from src_james.ensemble.util import Defensive_Copy


def findmodemap(A, B):
    A_array = np.array(A)
    B_pad = np.pad(B, ((1, 1), (1, 1)), "constant", constant_values=(-1, -1))
    A_pad = np.pad(A, ((1, 1), (1, 1)), "constant", constant_values=(-1, -1))
    m, n = A_pad.shape
    total_dict = {}
    A1 = A_pad.copy()
    A2 = A_pad.copy()
    for k in range(30):
        dict1 = {}
        for i in range(m):
            for j in range(n):

                if A1[i, j] != -1 and A1[i, j] != 0:
                    if str(A1[i - 1:i + 2, j - 1:j + 2]) not in dict1:
                        dict1[str(A1[i - 1:i + 2, j - 1:j + 2])] = B_pad[i - 1:i + 2, j - 1:j + 2]
                    if str(A1[i - 1:i + 2, j - 1:j + 2]) in dict1 and (
                            dict1[str(A1[i - 1:i + 2, j - 1:j + 2])] != B_pad[i - 1:i + 2, j - 1:j + 2]).any():
                        return -1

        total_dict = dict(dict1, **total_dict)
        # print(total_dict)
        A1_copy = A1.copy()
        # A1_copy=A1
        for i in range(m):
            for j in range(n):

                if str(A1_copy[i - 1:i + 2, j - 1:j + 2]) in total_dict.keys():
                    A2[i - 1:i + 2, j - 1:j + 2] = total_dict[str(A1_copy[i - 1:i + 2, j - 1:j + 2])]

        # plot_picture(A2.tolist())
        #         if (A1==B_pad).all():
        #             #print(k)
        #             break
        #         else:
        A1 = A2
        # plot_picture(A1.tolist())

    return total_dict


def usemodedict(A, total_dict):
    A_array = np.array(A)
    A_pad = np.pad(A, ((1, 1), (1, 1)), "constant", constant_values=(-1, -1))
    m, n = A_pad.shape
    total_dict = total_dict
    A1 = A_pad.copy()
    A2 = A_pad.copy()
    for k in range(100):
        A1_copy = A1.copy()
        for i in range(m):
            for j in range(n):
                if str(A1_copy[i - 1:i + 2, j - 1:j + 2]) in total_dict.keys():
                    # print(A1_copy[i-1:i+2,j-1:j+2])

                    A2[i - 1:i + 2, j - 1:j + 2] = total_dict[str(A1_copy[i - 1:i + 2, j - 1:j + 2])]

        A1 = A2
    return A2[1:-1, 1:-1].tolist()


def Solve_mode_dict(basic_task):
    Input  = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    Test_Case = Input[-1]
    Input = Input[:-1]
    total_dict = {}
    for i in range(10):
        mask = False
        for j in range(len(Input)):
            if i in Test_Case and i in Input[j]:
                mask = True
                break
            elif i not in Test_Case:
                mask = True
                break
        if mask == False:
            return -1

    for x, y in zip(Input, Output):
        if len(x) != len(y) or len(x[0]) != len(y[0]):
            return -1
        if findmodemap(x, y) == -1:
            return -1
        else:
            total_dict = dict(total_dict, **(findmodemap(x, y)))

    for x, y in zip(Input, Output):
        if y != usemodedict(x, total_dict):
            return -1

    return usemodedict(Test_Case, total_dict)

