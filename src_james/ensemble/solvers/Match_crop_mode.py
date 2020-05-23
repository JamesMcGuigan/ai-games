import numpy as np

from src_james.ensemble.crop import crop_mode, crop_mode_1, Apply_crop_mode
from src_james.ensemble.util import Defensive_Copy


def Match_crop_mode(basic_task):
    # returns -1 if no match is found
    # returns  Transformed_Test_Case  if the mathching rule is found
    Input = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    Test_Case = Input[-1]
    Input = Input[:-1]

    for i in range(len(crop_mode)):
        S = crop_mode[i]
        S1 = crop_mode_1[i]
        solved = True
        for x0, y in zip(Input, Output):
            x = np.array(x0)
            if Apply_crop_mode(S, x) == -1:
                solved = False
                break

            transformed_x = Apply_crop_mode(S, x)

            if transformed_x != y:
                solved = False
                break

        if solved == True:
            Transformed_Test_Case = Apply_crop_mode(S, Test_Case)
            return Transformed_Test_Case
    return -1


def Match_crop_mode_1(basic_task):
    # returns -1 if no match is found
    # returns  Transformed_Test_Case  if the mathching rule is found
    Input = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    Test_Case = Input[-1]
    Input = Input[:-1]

    for i in range(len(crop_mode)):
        S = crop_mode[i]
        S1 = crop_mode_1[i]
        solved = True
        for x0, y in zip(Input, Output):
            x = np.array(x0)
            if Apply_crop_mode(S, x) == -1:
                solved = False
                break

            transformed_x = Apply_crop_mode(S, x)
            #             plot_picture(transformed_x)
            transformed_x_array = np.array(transformed_x)
            transformed_x_1 = transformed_x_array[1:-1, 1:-1].tolist()
            #             plot_picture(transformed_x_1)

            if transformed_x_1 != y:
                solved = False
                break

        if solved == True:
            Transformed_Test_Case = Apply_crop_mode(S, Test_Case)
            Transformed_Test_Case_array = np.array(Transformed_Test_Case)
            Transformed_Test_Case_1 = Transformed_Test_Case_array[1:-1, 1:-1].tolist()
            return Transformed_Test_Case_1
    return -1
