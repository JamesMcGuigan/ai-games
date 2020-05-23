import numpy as np

from src_james.ensemble.util import Defensive_Copy


def get_period_length0(arr):
    H, W = arr.shape
    period = 1
    while True:
        cycled = np.pad(arr[:period, :], ((0, H - period), (0, 0)), 'wrap')
        if (cycled == arr).all():
            return period
        period += 1


def get_period_length1(arr):
    H, W = arr.shape
    period = 1
    while True:
        cycled = np.pad(arr[:, :period], ((0, 0), (0, W - period)), 'wrap')
        if (cycled == arr).all():
            return period
        period += 1


def get_period(arr0):
    if np.sum(arr0) == 0:
        return -1
    #     arr_crop=get_bound_image(arr0)
    #     arr=np.array(arr_crop)
    arr = np.array(arr0)
    a, b = get_period_length0(arr), get_period_length1(arr)
    period = arr[:a, :b]
    if period.shape == arr.shape:
        return -1
    return period.tolist()


def same_ratio(basic_task):
    # returns -1 if no match is found
    # returns Transformed_Test_Case  if the mathching rule is found
    # for this notebook we only look at mosaics
    Input  = [ Defensive_Copy(x) for x in basic_task[0] ]
    Output = [ Defensive_Copy(y) for y in basic_task[1] ]
    same_ratio = True
    R_x = []
    R_y = []
    for x, y in zip(Input[:-1], Output):

        if x == []:
            same_ratio = False
            break

        n1 = len(x)
        n2 = len(y)
        k1 = len(x[0])
        k2 = len(y[0])

        R_y.append(n2 / n1)
        R_x.append(k2 / k1)

    if same_ratio and min(R_x) == max(R_x) and min(R_y) == max(R_y):
        r1 = min(R_y)
        r2 = min(R_x)
        return r1, r2

    return -1
