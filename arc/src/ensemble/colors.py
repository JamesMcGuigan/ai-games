import numpy as np

from src.ensemble.util import Defensive_Copy



def maxcolor(A):
    b = np.bincount(A.flatten(), minlength=10)
    b[0] = 500  # ???255
    c = np.argsort(b)[-2]
    return c


# no count black
def mincolor(A):
    try:
        b = np.bincount(A.flatten(), minlength=10)
        b1 = np.delete(b, 0)
        # b1=b
        c = int(np.where(b == np.min(b1[np.nonzero(b1)]))[0])
        return c
    except:
        return 0


color_select = [maxcolor, mincolor, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def colorbycolor_select(A, S):
    A = np.array(Defensive_Copy(A))
    if S in color_select:
        if type(S) != int:
            x = S(A)
        else:
            x = S
        return x


def cropbycolor(a0, c):
    a = np.array(a0)
    if c not in a:
        return -1

    coords = np.argwhere(a == c)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    if x_min != x_max or y_min != y_max:
        return a[y_min:y_max + 1, x_min:x_max + 1], [y_min, y_max, x_min, x_max]
    else:
        return -1


def getObjectHash(pixmap):
    flat = pixmap.flatten().astype(np.bool)
    mult = np.array([2 ** x for x in range(len(flat))])
    return np.sum(flat * mult)


# 經測試沒有overflow問題
def groupByColor(pixmap):
    nb_colors = int(pixmap.max()) + 1
    splitted  = [(pixmap == i) * i for i in range(1, nb_colors)]
    return [x for x in splitted if np.any(x)]


def checkColorMap(a0, b0):
    a = np.array(a0)
    b = np.array(b0)
    a[a == 0] = 10
    b[b == 0] = 10
    c = 1
    inp_hashes = sorted([getObjectHash(pm) for pm in groupByColor(a)])
    out_hashes = sorted([getObjectHash(pm) for pm in groupByColor(b)])
    c *= inp_hashes == out_hashes
    return bool(c)


def findColorMap(a, b):
    colormap = {}
    a1 = np.array(a).flatten()
    b1 = np.array(b).flatten()

    for col, idx in zip(*np.unique(a1, return_index=True)):
        if col in colormap.keys(): continue
        colormap[col] = b1[idx]
    return colormap


def mergedict(dict1):
    dict3 = {}
    for dict2 in dict1:
        for key in dict2.keys():
            if key not in dict3.keys():
                dict3[key] = dict2[key]
            elif dict3[key] != dict2[key]:
                return False
    return dict3


def applyColorMap(pixmap, colormap):
    a1 = np.array(pixmap)
    for i in range(a1.shape[0]):
        for j in range(a1.shape[1]):
            if a1[i][j] not in colormap:  #
                continue  #
            a1[i][j] = colormap[pixmap[i][j]]
    return a1.tolist()


def object_dict(d):
    name_dic = {}
    obj_dic = {}
    for i in range(len(d)):
        d_obj = d[i]["obj"]
        if str(d_obj) not in name_dic:
            name_dic[str(d_obj)] = d_obj
            obj_dic[str(d_obj)] = 1
        else:
            obj_dic[str(d_obj)] += 1

    return obj_dic, name_dic


def object_01area_dict(d):
    name_dic = {}
    obj_dic = {}
    for i in range(len(d)):
        d_obj = d[i]["obj"]
        if str(d_obj) not in name_dic:
            name_dic[str(d_obj)] = d_obj

            obj_dic[str(d_obj)] = np.sum(np.where(np.array(d_obj), 1, 0))
    return obj_dic, name_dic


# color = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

def object_color_dict(d, c):
    name_dic = {}
    obj_dic = {}
    mask = False
    for i in range(len(d)):
        d_obj = d[i]["obj"]

        if c in np.array(d_obj) and mask == False:
            mask = True

        if str(d_obj) not in name_dic:
            name_dic[str(d_obj)] = d_obj
            obj_dic[str(d_obj)] = np.sum(np.where(np.array(d_obj) == c, 1, 0))

    if mask == True:
        return obj_dic, name_dic
    else:
        return False


def color_delete(a, b):
    c1 = np.unique(a)
    c2 = np.unique(b)
    # print(c1,c2)
    res = []
    if len(c2) > len(c1):
        return -1
    for i in range(len(c1)):
        if c1[i] not in c2:
            res.append(c1[i])
    #     if len(c2)+len(res)!=len(c1):
    #         return -1
    return res


def color_add_task(task):
    Input = task[0]
    Output = task[1]
    Test_Picture = Input[-1]
    Input = Input[:-1]
    if color_delete(Output[0], Input[0]) == -1:
        return -1
    del_c = set(color_delete(Output[0], Input[0]))
    for i in range(1, len(Input)):
        A = Input[i]
        B = Output[i]
        if color_delete(B, A) == -1:
            return -1
        else:
            del_c1 = set(color_delete(B, A))
        del_c = del_c & del_c1
    return list(del_c)


def filling(arr0, c):
    arr = np.array(arr0)
    y = Defensive_Copy(arr0)

    def get_closed_area(arr0):
        arr = np.array(arr0)
        # depth first search
        H, W = arr.shape
        Dy = [0, -1, 0, 1]
        Dx = [1, 0, -1, 0]
        arr_padded = np.pad(arr, ((1, 1), (1, 1)), "constant", constant_values=0)
        searched = np.zeros(arr_padded.shape, dtype=bool)
        searched[0, 0] = True
        q = [(0, 0)]
        while q:
            y, x = q.pop()
            for dy, dx in zip(Dy, Dx):
                y_, x_ = y + dy, x + dx
                if not 0 <= y_ < H + 2 or not 0 <= x_ < W + 2:
                    continue
                if not searched[y_][x_] and arr_padded[y_][x_] == 0:
                    q.append((y_, x_))
                    searched[y_, x_] = True
        res = searched[1:-1, 1:-1]
        res |= arr != 0
        return ~res

    arr[get_closed_area(y)] = c
    return arr.tolist()
