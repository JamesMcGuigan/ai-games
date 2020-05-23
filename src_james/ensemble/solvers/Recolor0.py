import numpy as np

from src_james.ensemble.split_object import get_bound_image


def Recolor0(task):
    Input = task[0]
    Output = task[1]
    Test_Picture = Input[-1]
    Input = Input[:-1]
    #     del_c_list=[]
    #     if color_delete_task(task)!=-1:
    #         del_c_list=color_delete_task(task)
    #         print(del_c_list)
    N = len(Input)

    x0 = Input[0]
    y0 = Output[0]
    n = len(x0)
    k = len(x0[0])
    a = len(y0)
    b = len(y0[0])
    for x in Input + [Test_Picture]:
        if len(x) != n or len(x[0]) != k:
            return -1
    for y in Output:
        if len(y) != a or len(y[0]) != b:
            return -1
    List1 = {}
    List2 = {}

    for i in range(n):
        for j in range(k):
            seq = []
            for x in Input:
                seq.append(x[i][j])
            List1[(i, j)] = seq

    for p in range(a):
        for q in range(b):
            seq1 = []
            for y in Output:
                seq1.append(y[p][q])

            places = []
            for key in List1:
                if List1[key] == seq1:
                    places.append(key)

            List2[(p, q)] = places
            if len(places) == 0:
                return -1

    answer = np.zeros((a, b), dtype=int)

    for p in range(a):
        for q in range(b):
            palette = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            for i, j in List2[(p, q)]:
                color = Test_Picture[i][j]
                palette[color] += 1
            #             for c in range(len(palette)):
            #                 if c in del_c_list:
            #                     palette[c]=0
            #             palette[0]=palette[0]//2

            answer[p, q] = np.argmax(palette)

    return answer.tolist()


def Recolor0_bound(task):
    Input0 = task[0]
    Output = task[1]
    Test_Picture0 = Input0[-1]
    Input0 = Input0[:-1]
    #     del_c_list=[]
    #     if color_delete_task(task)!=-1:
    #         del_c_list=color_delete_task(task)
    #         print(del_c_list)
    N = len(Input0)
    if len(np.unique(Test_Picture0)) == 1 and np.unique(Test_Picture0)[0] == 0:
        return -1
    Test_Picture = get_bound_image(Test_Picture0)
    Input = []
    for i in range(N):
        if len(np.unique(Input0[i])) == 1 and np.unique(Input0[i])[0] == 0:
            return -1
        else:
            Input.append(get_bound_image(Input0[i]))

    x0 = Input[0]
    y0 = Output[0]
    n = len(x0)
    k = len(x0[0])
    a = len(y0)
    b = len(y0[0])
    for x in Input + [Test_Picture]:
        if len(x) != n or len(x[0]) != k:
            return -1
    for y in Output:
        if len(y) != a or len(y[0]) != b:
            return -1
    List1 = {}
    List2 = {}

    for i in range(n):
        for j in range(k):
            seq = []
            for x in Input:
                seq.append(x[i][j])
            List1[(i, j)] = seq

    for p in range(a):
        for q in range(b):
            seq1 = []
            for y in Output:
                seq1.append(y[p][q])

            places = []
            for key in List1:
                if List1[key] == seq1:
                    places.append(key)

            List2[(p, q)] = places
            if len(places) == 0:
                return -1

    answer = np.zeros((a, b), dtype=int)

    for p in range(a):
        for q in range(b):
            palette = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            for i, j in List2[(p, q)]:
                color = Test_Picture[i][j]
                palette[color] += 1
            #             for c in range(len(palette)):
            #                 if c in del_c_list:
            #                     palette[c]=0
            #             palette[0]=palette[0]//4

            answer[p, q] = np.argmax(palette)

    return answer.tolist()
