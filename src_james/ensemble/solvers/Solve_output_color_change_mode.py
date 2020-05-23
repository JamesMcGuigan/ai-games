import numpy as np

from src_james.ensemble.colors import checkColorMap, applyColorMap
from src_james.ensemble.util import Defensive_Copy


def Match_output_color_change_mode(basic_task):
    # returns -1 if no match is found
    # returns  Transformed_Test_Case  if the mathching rule is found
    Input = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    Test_Case = Input[-1]
    Input = Input[:-1]
    A = np.array(Output[0])
    solved = True
    if np.array(Input[0]).shape != np.array(A).shape:
        return False
    for i in range(1, len(Output)):
        B = np.array(Output[i])
        if len(np.unique(B)) == 1:
            solved = False
            break
        if checkColorMap(A, B) == False:
            solved = False
            break
    return solved


def solve_output_color_change_mode(basic_task):
    # returns -1 if no match is found
    # returns  Transformed_Test_Case  if the mathching rule is found
    Input = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    Test_Case = Input[-1]
    Input = Input[:-1]
    if Match_output_color_change_mode(basic_task) == True:
        A = Output[0]
        use_c = -1
        for c in range(0, 10):
            res = True
            for x, y in zip(Input, Output):
                colormap = {}
                for i in range(len(x)):
                    for j in range(len(x[0])):
                        if x[i][j] != c and A[i][j] not in colormap:
                            colormap[A[i][j]] = x[i][j]

                B = applyColorMap(A, colormap)

                if B != y:
                    res = False
                    break

            if res == True:
                use_c = c
                break

        if use_c == -1:
            return -1
        else:
            colormap_T = {}
            for i in range(len(Test_Case)):
                for j in range(len(Test_Case[0])):
                    if Test_Case[i][j] != c and A[i][j] not in colormap_T:
                        colormap_T[A[i][j]] = Test_Case[i][j]

            B = applyColorMap(A, colormap_T)
            return B

    else:
        return -1
   