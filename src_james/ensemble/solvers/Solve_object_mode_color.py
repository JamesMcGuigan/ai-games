from src_james.ensemble.mode import max_color_mode, min_color_mode, mode_select
from src_james.ensemble.split_object import split_object801, split_object8, split_object01, split_object
from src_james.ensemble.util import Defensive_Copy

split_objects = [split_object, split_object01, split_object8, split_object801]


def sub_mode0_match(basic_task):
    Input = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    Test_Case = Input[-1]
    Input = Input[:-1]
    res = True
    for i in range(len(Input)):
        A = Input[i]
        B = Output[i]
        if len(split_object801(A)) == 1 or len(split_object801(B)) != 1:
            res = False
    return res


def object_mode_match(basic_task):
    Input = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    Test_Case = Input[-1]
    Input = Input[:-1]

    for m in range(len(split_objects)):
        for selectmode in mode_select:
            mode = True
            for i in range(len(Input)):
                A = Input[i]
                B = Output[i]
                A1 = split_objects[m](A)

                mode_image, name_dic = selectmode(A1)

                proposal_sol = name_dic[mode_image]
                # print(proposal_sol,B)
                if proposal_sol != B:
                    mode = False
            if mode == True:
                return split_objects[m], selectmode
    # print(mode)
    return mode


def solve_object_mode(basic_task):
    if object_mode_match(basic_task):
        split_object_mode, themode = object_mode_match(basic_task)
        # print(split_object_mode,themode)
    else:
        return -1
    Input = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    Test_Case = Input[-1]
    Input = Input[:-1]

    A1 = split_object_mode(Test_Case)
    mode_image, name_dic = themode(A1)

    return name_dic[mode_image]


def object_mode_color_match(basic_task):
    Input     = [ Defensive_Copy(x) for x in basic_task[0] ]
    Output    = [ Defensive_Copy(y) for y in basic_task[1] ]
    Test_Case = Input[-1]
    Input     = Input[:-1]

    for m in range(len(split_objects)):
        for c in range(0, 10):
            for maxmin in [max_color_mode, min_color_mode]:
                mode = True
                for i in range(len(Input)):
                    A = Input[i]
                    B = Output[i]
                    A1 = split_objects[m](A)
                    if max_color_mode(A1, c):
                        mode_image, name_dic = maxmin(A1, c)
                        # print(max_color_mode(A1,2))
                    else:
                        mode = False

                        break

                    proposal_sol = name_dic[mode_image]
                    # print(proposal_sol,B)
                    if proposal_sol != B:
                        mode = False
                if mode == True:
                    return split_objects[m], c, maxmin
    # print(mode)
    # return mode

def solve_object_mode_color(basic_task):
    if object_mode_color_match(basic_task):
        split_object_mode, c, maxmin_mode = object_mode_color_match(basic_task)
    else:
        return -1
    Input  = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    Test_Case = Input[-1]
    Input = Input[:-1]

    A1 = split_object_mode(Test_Case)
    mode_image, name_dic = maxmin_mode(A1, c)

    return name_dic[mode_image]
