import numpy as np

from src_james.ensemble.colors import color_select, colorbycolor_select
from src_james.ensemble.subarray import subarray_count
from src_james.ensemble.util import Defensive_Copy


def Match_color(basic_task):
    # returns -1 if no match is found
    # returns  Transformed_Test_Case  if the mathching rule is found
    Input  = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    Test_Case = Input[-1]
    Input = Input[:-1]
    for i in range(len(color_select)):
        S = color_select[i]
        solved = True

        for x, y in zip(Input, Output):
            c = colorbycolor_select(x, S)
            x1 = np.array(x)
            transformed_x = np.bincount(x1.flatten(), minlength=10)[c]
            if transformed_x == 0 or transformed_x != subarray_count(y, x):
                # 可擴充
                solved = False
                break
        if solved == True:
            Test_Case1 = np.array(Test_Case)
            Transformed_Test_Case = np.bincount(Test_Case1.flatten(), minlength=10)[colorbycolor_select(Test_Case, S)]
            return colorbycolor_select(Test_Case, S), Transformed_Test_Case

    return -1
# 顏色,數量
