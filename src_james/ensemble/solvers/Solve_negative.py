import numpy as np

from src_james.ensemble.color_classes import take_negative
from src_james.ensemble.colors import checkColorMap, findColorMap, mergedict, applyColorMap
from src_james.ensemble.split_object import get_bound_image
from src_james.ensemble.util import Defensive_Copy


def Solve_negative(basic_task):
    # returns -1 if no match is found
    # returns  Transformed_Test_Case  if the matching rule is found
    Input  = [ Defensive_Copy(x) for x in basic_task[0] ]
    Output = [ Defensive_Copy(y) for y in basic_task[1] ]
    Test_Case = Input[-1]
    Input     = Input[:-1]
    colormaps = {}

    for x0, y in zip(Input, Output):
        x = np.array(x0)
        x = get_bound_image(x)
        if take_negative(x) != -1:
            negative_x = take_negative(x)
        else:
            return -1

        if checkColorMap(negative_x, y) == False:
            return -1
        else:
            colormap = findColorMap(negative_x, y)

        if mergedict([colormaps, colormap]) == False:
            return -1

        colormaps  = mergedict([colormaps, colormap])
        pre_y_list = applyColorMap(negative_x, colormaps)

        if pre_y_list != y:
            return -1

    Test_Case_pred = applyColorMap(take_negative(take_negative(Test_Case)), colormaps)
    return Test_Case_pred
