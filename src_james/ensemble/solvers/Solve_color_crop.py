import numpy as np

from src_james.ensemble.colors import applyColorMap, mergedict
from src_james.ensemble.colors import color_select, colorbycolor_select, cropbycolor, checkColorMap, findColorMap
from src_james.ensemble.util import Defensive_Copy


def Solve_color_crop(basic_task):
    # returns -1 if no match is found
    # returns  Transformed_Test_Case  if the mathching rule is found
    Input  = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    Test_Case = Input[-1]
    Input = Input[:-1]
    for i in range(len(color_select)):
        solved = True
        colormaps = []

        for x, y0 in zip(Input, Output):
            if len(x) != len(y0) or len(x[0]) != len(y0[0]):
                return -1
            y = np.array(y0)
            c = colorbycolor_select(np.array(x), color_select[i])

            if cropbycolor(x, c) == -1:
                solved = False
                break
                # print(i,c)
            x_crop, locate = cropbycolor(x, c)
            # print(x_crop,locate)
            y_crop = y[locate[0]:locate[1] + 1, locate[2]:locate[3] + 1]
            # print(x_crop,y_crop)
            # print(checkColorMap(x_crop,y_crop))
            if checkColorMap(x_crop, y_crop):
                colormaps.append(findColorMap(x_crop, y_crop))
            else:
                solved = False
                break
                # print(findColorMap(x_crop,y_crop))
            # print(x_crop)
            # print(applyColorMap(x_crop,findColorMap(x_crop,y_crop)))
            if applyColorMap(x_crop, findColorMap(x_crop, y_crop)) == -1:
                solved = False
                break
            x_crop_color = applyColorMap(x_crop, findColorMap(x_crop, y_crop))

            x_array = np.array(x)
            x_array[locate[0]:locate[1] + 1, locate[2]:locate[3] + 1] = x_crop_color

            if not (x_array == y).all():
                return -1

        totalcolormap = mergedict(colormaps)
        # print(totalcolormap)
        if totalcolormap and solved == True:
            c = colorbycolor_select(np.array(Test_Case), color_select[i])
            if cropbycolor(Test_Case, c) == -1:
                continue

            Test_Case_crop, locate = cropbycolor(Test_Case, c)
            if applyColorMap(Test_Case_crop, totalcolormap) == -1:
                continue

            Test_Case_color = applyColorMap(Test_Case_crop, totalcolormap)
            # print(Test_Case_color)
            Test_Case_color0 = np.array(Test_Case_color)
            # print(locate)
            # print(Test_Case_color0)
            Test_Case_array = np.array(Test_Case)
            Test_Case_array[locate[0]:locate[1] + 1, locate[2]:locate[3] + 1] = Test_Case_color0
            return Test_Case_array.tolist()
    return -1
