import numpy as np

from src_james.ensemble.colors import checkColorMap, findColorMap, mergedict, applyColorMap
from src_james.ensemble.resize import resize_o, resize_c
from src_james.ensemble.split_object import get_bound_image
from src_james.ensemble.util import Defensive_Copy


def Solve_resize(basic_task):
    Input  = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    Test_Case = Input[-1]
    Input = Input[:-1]
    same_ratio = True
    colormaps  = {}
    R_x = []
    R_y = []
    r1, r2 = 0, 0
    for x, y in zip(Input, Output):
        if len(np.unique(x)) == 1 and np.unique(x)[0] == 0:
            return -1
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
    if same_ratio and min(R_x) == max(R_x) and min(R_y) == max(R_y):
        r1 = min(R_y)
        r2 = min(R_x)
    for x, y in zip(Input, Output):
        x_array = np.array(x)
        y_array = np.array(y)
        if r1 == 0 or r2 == 0:
            return -1
        pre_y = resize_o(x_array, r1, r2)
        if checkColorMap(pre_y, y_array) == False:
            return -1
        else:
            colormap = findColorMap(pre_y, y_array)
        if mergedict([colormaps, colormap]) == False:
            return -1
        colormaps = mergedict([colormaps, colormap])
        pre_y_list = applyColorMap(pre_y, colormaps)
        if pre_y_list != y:
            return -1

    Test_Case_array = np.array(Test_Case)
    return applyColorMap(resize_o(Test_Case_array, r1, r2), colormaps)


def Solve_resize_bound(basic_task):
    Input = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    Test_Case = Input[-1]
    Input = Input[:-1]
    same_ratio = True
    R_x = []
    R_y = []
    colormaps = {}
    r1, r2 = 0, 0
    for x0, y in zip(Input, Output):
        if len(np.unique(x0)) == 1 and np.unique(x0)[0] == 0:
            return -1

        x = get_bound_image(x0)
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
    if same_ratio and min(R_x) == max(R_x) and min(R_y) == max(R_y):
        r1 = min(R_y)
        r2 = min(R_x)
    for x0, y in zip(Input, Output):
        x = get_bound_image(x0)
        x_array = np.array(x)
        y_array = np.array(y)
        if r1 == 0 or r2 == 0:
            return -1
        pre_y = resize_o(x_array, r1, r2)
        if checkColorMap(pre_y, y_array) == False:
            return -1
        else:
            colormap = findColorMap(pre_y, y_array)
        if mergedict([colormaps, colormap]) == False:
            return -1
        colormaps = mergedict([colormaps, colormap])
        pre_y_list = applyColorMap(pre_y, colormaps)
        if pre_y_list != y:
            return -1

    Test_Case_array = np.array(get_bound_image(Test_Case))
    return applyColorMap(resize_o(Test_Case_array, r1, r2), colormaps)


def Solve_resizec(basic_task):
    Input = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    Test_Case = Input[-1]
    Input = Input[:-1]
    colormaps = {}
    for x, y in zip(Input, Output):
        if len(np.unique(x)) == 1 and np.unique(x)[0] == 0:
            return -1
        x_array = np.array(x)
        y_array = np.array(y)

        pre_y = resize_c(x_array)
        if pre_y.shape[0] > 30 or pre_y.shape[1] > 30:
            return -1

        if checkColorMap(pre_y, y_array) == False:
            return -1
        else:
            colormap = findColorMap(pre_y, y_array)
        if mergedict([colormaps, colormap]) == False:
            return -1
        colormaps = mergedict([colormaps, colormap])
        pre_y_list = applyColorMap(pre_y, colormaps)

        if np.shape(pre_y) != np.shape(y_array) or pre_y_list != y:
            return -1

    Test_Case_array = np.array(Test_Case)
    return applyColorMap(resize_c(Test_Case_array), colormaps)


def Solve_resizec_bound(basic_task):
    Input = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    Test_Case = Input[-1]
    Input = Input[:-1]
    colormaps = {}
    for x0, y in zip(Input, Output):
        if len(np.unique(x0)) == 1 and np.unique(x0)[0] == 0:
            return -1
        x = get_bound_image(x0)
        x_array = np.array(x)
        y_array = np.array(y)

        pre_y = resize_c(x_array)
        if pre_y.shape[0] > 30 or pre_y.shape[1] > 30:
            return -1

        if checkColorMap(pre_y, y_array) == False:
            return -1
        else:
            colormap = findColorMap(pre_y, y_array)
        if mergedict([colormaps, colormap]) == False:
            return -1
        colormaps = mergedict([colormaps, colormap])
        pre_y_list = applyColorMap(pre_y, colormaps)

        if np.shape(pre_y) != np.shape(y_array) or pre_y_list != y:
            return -1

    Test_Case_array = np.array(get_bound_image(Test_Case))
    return applyColorMap(resize_c(Test_Case_array), colormaps)
