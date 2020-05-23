import numpy as np

# 需要整合
from src_james.ensemble.colors import color_select, colorbycolor_select
from src_james.ensemble.split_object import get_bound_image
from src_james.ensemble.subarray import subarray_count
from src_james.ensemble.transformations import Glue
from src_james.ensemble.util import Defensive_Copy


def Match_color_bound(basic_task):
    # returns -1 if no match is found
    # returns  Transformed_Test_Case  if the mathching rule is found
    Input = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    Test_Case0 = Input[-1]  #####
    Test_Case = get_bound_image(Test_Case0)  ####
    Input = Input[:-1]
    for i in range(len(color_select)):
        S = color_select[i]
        solved = True

        for x0, y in zip(Input, Output):
            x = get_bound_image(x0)  ######
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


def Solve_mul_color_bound(basic_task):
    # returns -1 if no match is found
    # returns Transformed_Test_Case  if the mathching rule is found
    # for this notebook we only look at mul color match
    Input  = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    Test_Case0 = Input[-1]
    Test_Case = get_bound_image(Test_Case0)  ########################
    Input = Input[:-1]
    same_ratio = True
    R_x = []
    R_y = []
    for x0, y0 in zip(Input, Output):  ########
        x = get_bound_image(x0)  ########
        if x == []: return -1  #######
        y = y0
        n1 = len(x)
        n2 = len(y)
        k1 = len(x[0])
        k2 = len(y[0])
        if n2 % n1 != 0 or k2 % k1 != 0:
            same_ratio = False
            break
        else:
            R_y.append(n2 // n1)
            R_x.append(k2 // k1)

    if same_ratio and min(R_x) == max(R_x) and min(R_y) == max(R_y) and min(R_x) == len(Test_Case) and min(R_y) == len(
            Test_Case[0]):
        r1 = min(R_y)
        r2 = min(R_x)

        if Match_color_bound(basic_task) == -1:  ######
            return -1
        color, count = Match_color_bound(basic_task)  ############

        Test_Case_list    = []
        Test_Case_Wight_2 = []
        Test_Case_flatten = ((np.array(Test_Case)).flatten()).tolist()
        for panel in range(r1 * r2):
            Test_Case_list.append(Test_Case)

            if Test_Case_flatten[panel] != color:
                Test_Case_Wight_2.append(0)
            else:
                Test_Case_Wight_2.append(1)
        Partial_Solutions = []
        for panel in range(r1 * r2):
            Partial_Solutions.append((np.array(Test_Case_list[panel])) * Test_Case_Wight_2[panel])
        Transformed_Test_Case = Glue(Partial_Solutions, r1, r2)
        return Transformed_Test_Case

    return -1
