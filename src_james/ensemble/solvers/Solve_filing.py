from src_james.ensemble.colors import color_add_task, filling, checkColorMap, findColorMap, mergedict, applyColorMap
from src_james.ensemble.util import Defensive_Copy


def Solve_filling(basic_task):
    Input = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    Test_Case = Input[-1]
    Input = Input[:-1]
    colormaps = {}
    if color_add_task(basic_task) == -1 or len(color_add_task(basic_task)) < 1:
        return -1
    else:
        c = color_add_task(basic_task)[0]
    for x, y in zip(Input, Output):
        pre_y = filling(x, c)
        if checkColorMap(pre_y, y) == False:
            return -1
        else:
            colormap = findColorMap(pre_y, y)
        if mergedict([colormaps, colormap]) == False:
            return -1
        colormaps = mergedict([colormaps, colormap])
        pre_y = applyColorMap(pre_y, colormaps)
        if pre_y != y:
            return -1

    res = applyColorMap(filling(Test_Case, c), colormaps)
    return res
