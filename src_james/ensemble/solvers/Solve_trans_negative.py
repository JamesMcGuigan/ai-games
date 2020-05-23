from src_james.ensemble.color_classes import take_negative
from src_james.ensemble.colors import checkColorMap, findColorMap, mergedict, applyColorMap
from src_james.ensemble.transformations import Cut, Glue, Geometric, Apply_geometric
from src_james.ensemble.util import Defensive_Copy


def Match_trans_negative(basic_task):
    # returns -1 if no match is found
    # returns  Transformed_Test_Case  if the mathching rule is found
    Input = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    Test_Case = Input[-1]
    Input = Input[:-1]
    colormaps = {}
    colorchage = True
    for i in range(len(Geometric)):

        S = Geometric[i]
        solved = True
        for x, y in zip(Input, Output):
            x = take_negative(x)
            #             plot_picture(x)
            if x == -1:
                return -1
            transformed_x = Apply_geometric(S, x)
            #             plot_picture(transformed_x)
            #             plot_picture(y)
            #             print(checkColorMap(transformed_x,y))
            #             if transformed_x != y:
            #                 solved = False
            #                 break
            if checkColorMap(transformed_x, y) == False:
                solved = False
                break
            else:
                colormap = findColorMap(transformed_x, y)
                #                 print(transformed_x,y)
                #                 print(colormaps,colorchage)
                if colorchage == True:
                    colormaps = mergedict([colormap, colormaps])
                if colormaps == False:
                    colorchage = False

        if solved == True:
            if take_negative(Test_Case) == -1:
                return -1
            if colorchage == True:
                Transformed_Test_Case = applyColorMap(Apply_geometric(S, take_negative(Test_Case)), colormaps)
            if colorchage == False:
                Transformed_Test_Case = Apply_geometric(S, Test_Case)
            return Transformed_Test_Case
    return -1


def Solve_trans_negative(basic_task):
    # returns -1 if no match is found
    # returns Transformed_Test_Case  if the mathching rule is found
    # for this notebook we only look at mosaics
    Input  = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    same_ratio = True
    R_x = []
    R_y = []

    for x, y in zip(Input[:-1], Output):
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
        Fractured_Output = [Cut(x, r1, r2) for x in Output]

        Partial_Solutions = []
        for panel in range(r1 * r2):
            List = [Fractured_Output[i][panel] for i in range(len(Output))]
            proposed_solution = Match_trans_negative([Input, List])

            if proposed_solution == -1:
                return -1
            else:
                Partial_Solutions.append(proposed_solution)
        Transformed_Test_Case = Glue(Partial_Solutions, r1, r2)
        return Transformed_Test_Case

    return -1
