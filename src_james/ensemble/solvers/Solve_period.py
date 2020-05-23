import numpy as np

from src_james.ensemble.colors import mergedict, applyColorMap, checkColorMap, findColorMap
from src_james.ensemble.period import get_period, same_ratio
from src_james.ensemble.util import Defensive_Copy


def Solve_period(basic_task):
    # returns -1 if no match is found
    # returns Transformed_Test_Case  if the mathching rule is found
    # for this notebook we only look at mosaics
    Input  = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    Test_Case = Input[-1]
    Input     = Input[:-1]
    colormaps = {}
    if same_ratio(basic_task) == -1:
        return -1
    else:
        R_y, R_X = same_ratio(basic_task)

    for x, y in zip(Input[:-1], Output):

        if get_period(x) == -1:
            return -1
        period_image = get_period(x)
        y_shape = np.zeros((int(len(x) * R_y), int(len(x[0]) * R_X)))
        if len(y_shape) < len(period_image) or len(y_shape[0]) < len(period_image[0]):
            return -1
        y_pred = np.pad(period_image,
                        ((0, len(y_shape) - len(period_image)), (0, len(y_shape[0]) - len(period_image[0]))), "wrap")

        if checkColorMap(y_pred, y):
            colormap = findColorMap(y_pred, y)

            if mergedict([colormaps, colormap]) == False:
                return -1
            colormaps = mergedict([colormaps, colormap])
        else:
            return -1
    if colormaps:
        period_image = get_period(Test_Case)
        y_shape = np.zeros((int(len(Test_Case) * R_y), int(len(Test_Case[0]) * R_X)))
        y_pred  = np.pad(period_image,
                        ((0, len(y_shape) - len(period_image)), (0, len(y_shape[0]) - len(period_image[0]))), "wrap")
        y_pred_final = applyColorMap(y_pred, colormaps)
        return y_pred_final
    else:
        return -1
           