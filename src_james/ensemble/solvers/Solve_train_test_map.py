from src_james.ensemble.colors import checkColorMap, findColorMap, applyColorMap
from src_james.ensemble.util import Defensive_Copy


def Solve_train_test_map(basic_task):
    # returns -1 if no match is found
    # returns Transformed_Test_Case  if the mathching rule is found
    # for this notebook we only look at mosaics
    Input  = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    Test_Case = Input[-1]
    Input = Input[:-1]
    for x, y in zip(Input, Output):
        if checkColorMap(x,Test_Case)==True:
            colomap = findColorMap(x,Test_Case)
            return applyColorMap(y,colomap)
    return -1