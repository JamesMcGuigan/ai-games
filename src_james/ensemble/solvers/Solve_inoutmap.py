# plit_object801
import numpy as np

from src_james.ensemble.colors import checkColorMap, findColorMap, applyColorMap
from src_james.ensemble.split_object import split_object801
from src_james.ensemble.util import Defensive_Copy


def inoutmap(basic_task, n1, n2, m1, m2):
    Input  = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    Test_Case = Input[-1]
    Input     = Input[:-1]
    name_dic  = {}
    obj_dic   = {}
    for x, y in zip(Input, Output):
        x_array = np.array(x)
        y_array = np.array(y)
        n0, m0 = len(x), len(x[0])
        a = split_object801(x)
        if len(a) > 10 or len(a) == 0:
            return -1
        obj = []
        for i in range(len(a)):
            obj.append(a[i]["obj"])
        di_obj = []
        for i in range(len(obj)):
            if obj[i] not in di_obj:
                di_obj.append(obj[i])

        for k in range(len(di_obj)):
            example = di_obj[k]
            # print(example)
            n, m = len(example), len(example[0])
            for i in range(n0 - n + 1):
                for j in range(m0 - m + 1):
                    if x_array[i:i + n, j:j + m].tolist() == example:
                        if i - n1 >= 0 and i + n + n2 <= n0 and j - m1 >= 0 and j + m + m2 <= m0:
                            tmp_x = x_array[i - n1:i + n + n2, j - m1:j + m + m2]
                            tmp_y = y_array[i - n1:i + n + n2, j - m1:j + m + m2]
                            if str(tmp_x) not in obj_dic:
                                name_dic[str(tmp_x)] = tmp_x
                                obj_dic[str(tmp_x)] = tmp_y
        # print(obj_dic)

    return name_dic, obj_dic


def solve_inoutmap(basic_task, n1, n2, m1, m2):
    Input  = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    Test_Case = Input[-1]
    Input = Input[:-1]

    if inoutmap(basic_task, n1, n2, m1, m2) == -1:
        return -1
    name_dic, object_dict = inoutmap(basic_task, n1, n2, m1, m2)

    for x, y in zip(Input, Output):

        if len(x) != len(y) or len(x[0]) != len(y[0]):
            return -1
        x_array      = np.array(x)
        x_array_pad  = np.pad(x_array, ((n1, n2), (m1, m2)), 'constant', constant_values=(0, 0))
        x_array_copy = x_array.copy()
        x_pad1       = np.pad(x_array_copy, ((n1, n2), (m1, m2)), 'constant', constant_values=(0, 0))
        y_array      = np.array(y)
        n0, m0 = len(x), len(x[0])

        a = split_object801(x)

        if len(a) > 10 or len(a) == 0:
            return -1
        obj = []
        for i in range(len(a)):
            obj.append(a[i]["obj"])
        di_obj = []
        for i in range(len(obj)):
            if obj[i] not in di_obj:
                di_obj.append(obj[i])
        # print(di_obj)
        for k in range(len(di_obj)):
            example = di_obj[k]
            n, m = len(example), len(example[0])
            for i in range(n0 - n + 1):
                for j in range(m0 - m + 1):
                    if x_array_pad[i + n1:i + n + n1, j + m1:j + m + m1].tolist() == example:

                        if str(x_array_pad[i:i + n + n1 + n2, j:j + m + m1 + m2]) not in object_dict:
                            return -1

                        try:
                            x_pad1[i:i + n + n1 + n2, j:j + m + m1 + m2] = object_dict[
                                str(x_array_pad[i:i + n + n1 + n2, j:j + m + m1 + m2])]

                        except:
                            return -1

        # avoid x_pad1[0:0,0:0]
        # print(x_pad1)
        x_pad2 = np.pad(x_pad1, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))
        # print(x_pad2)
        res = x_pad2[1 + n1:-1 - n2, 1 + m1:-1 - m2]
        # print(res)

        if not (res == y_array).all():
            return -1

    Test_Case_array = np.array(Test_Case)
    Test_Case_array_copy = Test_Case_array.copy()
    Test_Case_pad = np.pad(Test_Case_array_copy, ((n1, n2), (m1, m2)), 'constant', constant_values=(0, 0))
    Test_Case_pad1 = np.pad(Test_Case_array_copy, ((n1, n2), (m1, m2)), 'constant', constant_values=(0, 0))
    n0, m0 = len(Test_Case), len(Test_Case[0])
    a = split_object801(Test_Case)
    obj = []

    for i in range(len(a)):
        obj.append(a[i]["obj"])
    di_obj = []
    for i in range(len(obj)):
        if obj[i] not in di_obj:
            di_obj.append(obj[i])
    for k in range(len(di_obj)):
        example = di_obj[k]
        n, m = len(example), len(example[0])
        for i in range(n0 - n + 1):
            for j in range(m0 - m + 1):
                if Test_Case_pad[i + n1:i + n + n1, j + m1:j + m + m1].tolist() == example:

                    if str(Test_Case_pad[i:i + n + n1 + n2, j:j + m + m1 + m2]) not in object_dict:
                        return -1

                    Test_Case_pad1[i:i + n + n1 + n2, j:j + m + m1 + m2] = object_dict[
                        str(Test_Case_pad[i:i + n + n1 + n2, j:j + m + m1 + m2])]
                    Test_Case_pad2 = np.pad(Test_Case_pad1, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))

    return Test_Case_pad2[1 + n1:-1 - n2, 1 + m1:-1 - m2].tolist()


def solve_inoutmap_colormap(basic_task, n1, n2, m1, m2):
    Input  = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    Test_Case = Input[-1]
    Input = Input[:-1]

    if inoutmap(basic_task, n1, n2, m1, m2) == -1:
        return -1
    name_dic, object_dict = inoutmap(basic_task, n1, n2, m1, m2)

    for x, y in zip(Input, Output):

        if len(x) != len(y) or len(x[0]) != len(y[0]):
            return -1
        x_array = np.array(x)
        x_array_pad = np.pad(x_array, ((n1, n2), (m1, m2)), 'constant', constant_values=(0, 0))
        # print(x_array_pad)
        x_array_copy = x_array.copy()
        x_pad1 = np.pad(x_array_copy, ((n1, n2), (m1, m2)), 'constant', constant_values=(0, 0))
        y_array = np.array(y)
        n0, m0 = len(x), len(x[0])

        a = split_object801(x)

        if len(a) > 10 or len(a) == 0:
            return -1
        obj = []
        for i in range(len(a)):
            obj.append(a[i]["obj"])
        di_obj = []
        for i in range(len(obj)):
            if obj[i] not in di_obj:
                di_obj.append(obj[i])
        # print(di_obj)
        for k in range(len(di_obj)):
            example = di_obj[k]
            n, m = len(example), len(example[0])
            for i in range(n0 - n + 1):
                for j in range(m0 - m + 1):

                    if x_array_pad[i + n1:i + n + n1, j + m1:j + m + m1].tolist() == example:

                        if str(x_array_pad[i:i + n + n1 + n2, j:j + m + m1 + m2]) in object_dict:

                            x_pad1[i:i + n + n1 + n2, j:j + m + m1 + m2] = object_dict[
                                str(x_array_pad[i:i + n + n1 + n2, j:j + m + m1 + m2])]
                        else:
                            color_res = False
                            for value in name_dic.values():
                                if checkColorMap(x_array_pad[i:i + n + n1 + n2, j:j + m + m1 + m2], value) == True:
                                    colormap = findColorMap(value, x_array_pad[i:i + n + n1 + n2, j:j + m + m1 + m2])

                                    x_pad1[i:i + n + n1 + n2, j:j + m + m1 + m2] = \
                                        np.array(applyColorMap(object_dict[str(value)], colormap))
                                    color_res = True
                                    break
                            if color_res == False:
                                return -1

        # avoid x_pad1[0:0,0:0]
        # print(x_pad1)
        x_pad2 = np.pad(x_pad1, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))

        # print(x_pad2)
        res = x_pad2[1 + n1:-1 - n2, 1 + m1:-1 - m2]

        if not (res == y_array).all():
            return -1

    Test_Case_array = np.array(Test_Case)
    Test_Case_array_copy = Test_Case_array.copy()
    Test_Case_pad  = np.pad(Test_Case_array_copy, ((n1, n2), (m1, m2)), 'constant', constant_values=(0, 0))
    Test_Case_pad1 = np.pad(Test_Case_array_copy, ((n1, n2), (m1, m2)), 'constant', constant_values=(0, 0))
    n0, m0 = len(Test_Case), len(Test_Case[0])
    a = split_object801(Test_Case)
    obj = []

    for i in range(len(a)):
        obj.append(a[i]["obj"])

    di_obj = []
    for i in range(len(obj)):
        if obj[i] not in di_obj:
            di_obj.append(obj[i])

    for k in range(len(di_obj)):
        example = di_obj[k]
        n, m = len(example), len(example[0])
        for i in range(n0 - n + 1):
            for j in range(m0 - m + 1):
                if Test_Case_pad[i + n1:i + n + n1, j + m1:j + m + m1].tolist() == example:

                    if str(Test_Case_pad[i:i + n + n1 + n2, j:j + m + m1 + m2]) in object_dict:
                        Test_Case_pad1[i:i + n + n1 + n2, j:j + m + m1 + m2] = object_dict[
                            str(Test_Case_pad[i:i + n + n1 + n2, j:j + m + m1 + m2])]
                    else:
                        color_res = False
                        for value in name_dic.values():
                            if checkColorMap(Test_Case_pad[i:i + n + n1 + n2, j:j + m + m1 + m2], value) == True:
                                colormap = findColorMap(value, Test_Case_pad[i:i + n + n1 + n2, j:j + m + m1 + m2])
                                Test_Case_pad1[i:i + n + n1 + n2, j:j + m + m1 + m2] = np.array(
                                    applyColorMap(object_dict[str(value)], colormap))
                                color_res = True
                                break
                        if color_res == False:
                            return -1

                    Test_Case_pad2 = np.pad(Test_Case_pad1, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))

    return Test_Case_pad2[1 + n1:-1 - n2, 1 + m1:-1 - m2].tolist()
