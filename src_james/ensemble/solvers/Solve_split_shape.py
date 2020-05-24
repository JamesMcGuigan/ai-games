import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

from src_james.ensemble.sample_sub.path import data_path, training_path, evaluation_path

for dirname, _, filenames in os.walk('/kaggle/input'):
    print(dirname)

test_path      = data_path / 'test'
training_tasks = sorted(os.listdir(training_path))
eval_tasks     = sorted(os.listdir(evaluation_path))

T = training_tasks
Trains = []
for i in range(400):
    task_file = str(training_path / T[i])
    task = json.load(open(task_file, 'r'))
    Trains.append(task)

E = eval_tasks
Evals = []
for i in range(400):
    task_file = str(evaluation_path / E[i])
    task = json.load(open(task_file, 'r'))
    Evals.append(task)

cmap = colors.ListedColormap(
    ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)
# 0:black, 1:blue, 2:red, 3:greed, 4:yellow,
# 5:gray, 6:magenta, 7:orange, 8:sky, 9:brown
plt.figure(figsize=(5, 2), dpi=200)
plt.imshow([list(range(10))], cmap=cmap, norm=norm)
plt.xticks(list(range(10)))
plt.yticks([])
plt.show()


def plot_task(task):
    n = len(task["train"]) + len(task["test"])
    fig, axs = plt.subplots(2, n, figsize=(4 * n, 8), dpi=50)
    plt.subplots_adjust(wspace=0, hspace=0)
    fig_num = 0

    for i, t in enumerate(task["train"]):
        t_in, t_out = np.array(t["input"]), np.array(t["output"])
        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)
        axs[0][fig_num].set_title(f'Train-{i} in')
        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))
        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))
        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)
        axs[1][fig_num].set_title(f'Train-{i} out')
        axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))
        axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))
        fig_num += 1
    for i, t in enumerate(task["test"]):
        t_in, t_out = np.array(t["input"]), np.array(t["output"])
        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)
        axs[0][fig_num].set_title(f'Test-{i} in')
        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))
        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))
        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)
        axs[1][fig_num].set_title(f'Test-{i} out')
        axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))
        axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))
        fig_num += 1

    plt.tight_layout()
    plt.show()


def plot_picture(x):
    plt.imshow(np.array(x), cmap=cmap, norm=norm)
    plt.show()


def Defensive_Copy(A):
    if type(A) != list:
        A = A.tolist()
    n = len(A)
    k = len(A[0])
    L = np.zeros((n, k), dtype=int)
    for i in range(n):
        for j in range(k):
            L[i, j] = 0 + A[i][j]
    return L.tolist()


def Create(task, task_id=0):
    n = len(task['train'])
    Input = [Defensive_Copy(task['train'][i]['input']) for i in range(n)]
    Output = [Defensive_Copy(task['train'][i]['output']) for i in range(n)]
    Input.append(Defensive_Copy(task['test'][task_id]['input']))
    return Input, Output


def getObjectHash(pixmap):
    flat = pixmap.flatten().astype(np.bool)
    mult = np.array([2 ** x for x in range(len(flat))])
    return np.sum(flat * mult)


# 經測試沒有overflow問題
def groupByColor(pixmap):
    nb_colors = int(pixmap.max()) + 1
    splited = [(pixmap == i) * i for i in range(1, nb_colors)]
    return [x for x in splited if np.any(x)]


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


def applyColorMap(pixmap, colormap):
    a1 = np.array(pixmap)
    for i in range(a1.shape[0]):
        for j in range(a1.shape[1]):
            if a1[i][j] not in colormap:  #
                continue  #
            a1[i][j] = colormap[pixmap[i][j]]
    return a1.tolist()


def mergedict(dict1):
    dict3 = {}
    for dict2 in dict1:
        for key in dict2.keys():
            if key not in dict3.keys():
                dict3[key] = dict2[key]
            elif dict3[key] != dict2[key]:
                return False
    return dict3


def color_classes(a):
    b = len(np.nonzero(np.unique(a))[0])
    return b


# color classes (not include 0)
def take_negative(a0):
    a = np.array(a0)
    a_copy = a.copy()
    if color_classes(a) == 2:
        if 0 in a:
            c1, c2 = np.unique(a)[1], np.unique(a)[2]
        else:
            c1, c2 = np.unique(a)[0], np.unique(a)[1]
        for i in range(len(a0)):
            for j in range(len(a0[0])):
                if a[i][j] == c1:
                    a_copy[i][j] = c2
                elif a[i][j] == c2:
                    a_copy[i][j] = c1

        return a_copy.tolist()
    elif color_classes(a) == 1:
        if 0 in a:
            c1, c2 = np.unique(a)[0], np.unique(a)[1]
        else:
            return -1
        for i in range(len(a0)):
            for j in range(len(a0[0])):
                if a[i][j] == c1:
                    a_copy[i][j] = c2
                elif a[i][j] == c2:
                    a_copy[i][j] = c1

        return a_copy.tolist()

    else:
        return -1


# Transformations
def Vert(M):
    n = len(M)
    k = len(M[0])
    ans = np.zeros((n, k), dtype=int)
    for i in range(n):
        for j in range(k):
            ans[i][j] = 0 + M[n - 1 - i][j]
    return ans.tolist()


def Hor(M):
    n = len(M)
    k = len(M[0])
    ans = np.zeros((n, k), dtype=int)
    for i in range(n):
        for j in range(k):
            ans[i][j] = 0 + M[i][k - 1 - j]
    return ans.tolist()


def Rot1(M):
    n = len(M)
    k = len(M[0])
    ans = np.zeros((k, n), dtype=int)
    for i in range(n):
        for j in range(k):
            ans[j][i] = 0 + M[i][k - 1 - j]
    return ans.tolist()


def Rot2(M):
    n = len(M)
    k = len(M[0])
    ans = np.zeros((k, n), dtype=int)
    for i in range(n):
        for j in range(k):
            ans[j][i] = 0 + M[n - 1 - i][j]
    return ans.tolist()


Geometric = [[Hor, Hor], [Rot2], [Rot1, Rot1], [Rot1], [Vert], [Hor, Rot2], [Hor], [Vert, Rot2]]

BACKGROUND = 0


def _get_bound(img0):
    BACKGROUND = 0  #
    img = np.array(img0)
    h, w = img.shape
    x0 = w - 1
    x1 = 0
    y0 = h - 1
    y1 = 0
    for x in range(w):
        for y in range(h):
            if img[y, x] == BACKGROUND:
                continue
            x0 = min(x0, x)
            x1 = max(x1, x)
            y0 = min(y0, y)
            y1 = max(y1, y)
    return x0, x1, y0, y1


def get_bound_image(img0):
    BACKGROUND = 0  #
    x0, x1, y0, y1 = _get_bound(img0)
    img = np.array(img0)
    return img[y0:y1 + 1, x0:x1 + 1].tolist()


def Apply_geometric(S, x):
    if S in Geometric:
        x1 = Defensive_Copy(x)
        for t in S:
            x1 = t(x1)
    return x1


def Cut(M, r1, r2):  # Cut a region into tiles
    List = []
    n = len(M)
    n1 = n // r1
    k = len(M[0])
    k1 = k // r2
    for i in range(r1):
        for j in range(r2):
            R = np.zeros((n1, k1), dtype=int)
            for t1 in range(n1):
                for t2 in range(k1):
                    R[t1, t2] = 0 + M[i * n1 + t1][j * k1 + t2]
            List.append(R.tolist())
    return List


def Glue(List, r1, r2):  # Combine tiles to one picture
    n1 = len(List[0])
    k1 = len(List[0][0])
    ans = np.zeros((n1 * r1, k1 * r2), dtype=int)
    counter = 0
    for i in range(r1):
        for j in range(r2):
            R = List[counter]
            counter += 1
            for t1 in range(n1):
                for t2 in range(k1):
                    ans[i * n1 + t1, j * k1 + t2] = 0 + R[t1][t2]
    return ans.tolist()


def issubarray(A0, B0):
    A = np.array(A0)
    B = np.array(B0)
    a1 = A.shape[0]
    a2 = A.shape[1]
    b1 = B.shape[0]
    b2 = B.shape[1]
    if (a1 == b1 and a2 == b2) or b1 > a1 or b2 > a2 or (b1 == 1 and b2 == 1):
        return False
    c1 = a1 - b1 + 1
    c2 = a2 - b2 + 1
    for i in range(c1):
        for j in range(c2):
            if (A[i:i + b1, j:j + b2] == B).all():
                return True
    return False


def issubarray_match(task):
    for i in range(len(task)):
        A = np.array(task[i]["input"])
        B = np.array(task[i]["output"])
        if issubarray(A, B) == False:
            return False
    return True


def subarray_count(A, B):
    A = np.array(A)
    B = np.array(B)
    a1 = A.shape[0]
    a2 = A.shape[1]
    b1 = B.shape[0]
    b2 = B.shape[1]
    if (a1 == b1 and a2 == b2) or b1 > a1 or b2 > a2 or (b1 == 1 and b2 == 1):
        return 0
    c1 = a1 - b1 + 1
    c2 = a2 - b2 + 1
    i = 0
    j = 0
    count = 0
    t = 0
    while i < c1:
        while j < c2:
            if (A[i:i + b1, j:j + b2] == B).all():
                count += 1
                j += b1
                t = 1
            else:
                j += 1
        j = 0
        if t == 1:
            i += b2
        else:
            i += 1
    return count


# A比B大
def color_counts(a):
    a = np.array(a)
    b = np.bincount(a.flatten(), minlength=10)
    b = b[1:]
    return b


# 去黑
def color_counts_mul(C, E):
    c1 = color_counts(C)
    e1 = color_counts(E)
    m = 0
    for i in range(len(e1)):
        if c1[i] == 0 and e1[i] == 0:
            continue
        elif c1[i] < e1[i] or (c1[i] == 0 and e1[i] != 0) or (e1[i] == 0 and c1[i] != 0):
            return 0
        else:
            if c1[i] % e1[i] != 0:
                return 0
            else:
                n = c1[i] // e1[i]
                if m != 0 and m != n:
                    return 0
                m = n
    return m


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


# np.bincount(A)[color_select]

def colorbycolor_select(A, S):
    A = np.array(Defensive_Copy(A))
    if S in color_select:
        if type(S) != int:
            x = S(A)
        else:
            x = S
        return x


# A:pattern
def is_juxt(A, B):
    background = 0  #
    if -1 in B:
        return 0
    if len(np.unique(B)) == 1 and np.unique(B)[0] == 0:
        return 0
    n = len(A)
    m = len(A[0])
    if len(B) != n or len(B[0]) != m:
        return 0
    count = 0
    for i in range(n):
        for j in range(m):
            if A[i][j] == background and B[i][j] != background:
                return 0
            elif A[i][j] != background and B[i][j] != background and B[i][j] != A[i][j]:
                return 0
            elif B[i][j] == A[i][j]:
                count += 1

    return count / (m * n)


def juxt_pattern_to_image0(pattern, image):
    image_array = np.pad(image, ((1, 1), (1, 1)), "constant", constant_values=0)
    image_array_copy = image_array.copy()
    pattern_array = np.array(pattern)
    a = len(pattern)
    b = len(pattern[0])
    m, n = len(image), len(image[0])

    #     def min_req(pattern,image_part):
    #         res=[]
    #         b=list(np.nonzero(np.unique(image_part))[0])
    #         for i in range(1,len(color_counts(pattern))):
    #             if  i in b:
    #                 res.append(color_counts(pattern)[i])
    #         if len(res)==0:
    #             return -1

    #         return min(res)

    while True:
        max_pro = 0
        for i in range(1, m - a + 1 + 1):
            for j in range(1, n - b + 1 + 1):
                tmp_image = image_array[i:i + a, j:j + b].tolist()
                if np.sum(image_array[i:i + a, j:j + b]) != np.sum(image_array[i - 1:i + a + 2, j - 1:j + b + 2]):
                    continue

                if is_juxt(pattern, tmp_image) != 0:
                    max_pro = max(max_pro, is_juxt(pattern, tmp_image))

        #         print(image_array)
        #         print(image_array_copy)
        if max_pro == 0:
            return image_array_copy[1:-1, 1:-1].tolist()

        for i in range(1, m - a + 1 + 1):
            for j in range(1, n - b + 1 + 1):
                tmp_image = image_array[i:i + a, j:j + b].tolist()
                if is_juxt(pattern, tmp_image) == max_pro:
                    image_array_copy[i:i + a, j:j + b] = pattern_array

                    image_array[i:i + a, j:j + b] = -1


_neighbor_offsets = {
    4: [(1, 0), (-1, 0), (0, 1), (0, -1)],
    8: [(1, 0), (-1, 0), (0, 1), (0, -1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
}


def _expand_region_indices(img, i, j, neighbor=4):
    h, w = img.shape
    seed_color = img[i, j]
    idx = np.zeros_like(img, dtype=np.bool)
    region = []
    region.append((i, j))
    while len(region) > 0:
        ii, jj = region.pop()
        if img[ii, jj] != seed_color:
            continue
        idx[ii, jj] = True
        for di, dj in _neighbor_offsets[neighbor]:
            ni, nj = ii + di, jj + dj
            if ni >= 0 and ni < h and nj >= 0 and nj < w \
                    and not idx[ni, nj]:
                region.append((ni, nj))
    return idx


def _expand_region_indices01(img, i, j, neighbor=4):
    h, w = img.shape
    seed_color = 1
    idx = np.zeros_like(img, dtype=np.bool)
    region = []
    region.append((i, j))
    while len(region) > 0:
        ii, jj = region.pop()
        if img[ii, jj] == 0:
            continue
        idx[ii, jj] = True
        for di, dj in _neighbor_offsets[neighbor]:
            ni, nj = ii + di, jj + dj
            if ni >= 0 and ni < h and nj >= 0 and nj < w \
                    and not idx[ni, nj]:
                region.append((ni, nj))
    return idx


def _split_conn(img0, neighbor=4):
    regions = []
    img = np.array(img0)
    mem = np.zeros_like(img, dtype=np.bool)
    h, w = img.shape
    for j in range(w):
        for i in range(h):
            p = img[i, j]
            if p == BACKGROUND or mem[i, j]:
                continue
            conn_idx = _expand_region_indices(img, i, j, neighbor)
            mem[conn_idx] = True
            regions.append((np.where(conn_idx, img, BACKGROUND)).tolist())
    return regions


def _split_conn01(img0, neighbor=4):
    regions = []
    img = np.array(img0)
    mem = np.zeros_like(img, dtype=np.bool)
    h, w = img.shape
    for j in range(w):
        for i in range(h):
            p = img[i, j]
            if p == BACKGROUND or mem[i, j]:
                continue
            conn_idx = _expand_region_indices01(img, i, j, neighbor)
            mem[conn_idx] = True
            regions.append((np.where(conn_idx, img, BACKGROUND)).tolist())
    return regions


def split_conn(img):
    ''' Split an image into a list of images each containing a single connected region'''

    return _split_conn(img, 4)


def split_conn8(img):
    ''' Split an image into a list of images each containing a single connected region.
      Pixels of 8 neighbors are all considered "connected"
    '''

    return _split_conn(img, 8)


def split_conn01(img):
    ''' Split an image into a list of images each containing a single connected region'''

    return _split_conn01(img, 4)


def split_conn801(img):
    ''' Split an image into a list of images each containing a single connected region.
      Pixels of 8 neighbors are all considered "connected"
    '''

    return _split_conn01(img, 8)


def _split_object(img0, neighbor=4):
    regions = []
    img = np.array(img0)
    mem = np.zeros_like(img, dtype=np.bool)
    h, w = img.shape
    for j in range(w):
        for i in range(h):
            p = img[i, j]
            if p == BACKGROUND or mem[i, j]:
                continue
            conn_idx = _expand_region_indices(img, i, j, neighbor)
            mem[conn_idx] = True
            splitimage = np.where(conn_idx, img, BACKGROUND)
            #             minx=0
            #             miny=0
            #             maxx=0
            #             maxy=0
            #             for m in range(h):
            #                 if sum(splitimage[m,:])!=0:
            #                     miny=m
            #                     break
            #             for n in range(w):
            #                 if sum(splitimage[:,n])!=0:
            #                     minx=n
            #                     break
            #             for m in range(h-1,-1,-1):
            #                 if sum(splitimage[m,:])!=0:
            #                     maxy=m
            #                     break
            #             for n in range(w-1,-1,-1):
            #                 if sum(splitimage[:,n])!=0:
            #                     maxx=n
            #                     break
            (minx, maxx, miny, maxy) = _get_bound(splitimage)
            split_object = (splitimage[miny:maxy + 1, minx:maxx + 1]).tolist()

            regions.append({'start': (miny, minx), 'obj': split_object})
    return regions


def _split_object01(img0, neighbor=4):
    regions = []
    img = np.array(img0)
    mem = np.zeros_like(img, dtype=np.bool)
    h, w = img.shape
    for j in range(w):
        for i in range(h):
            p = img[i, j]
            if p == BACKGROUND or mem[i, j]:
                continue
            conn_idx = _expand_region_indices01(img, i, j, neighbor)
            mem[conn_idx] = True
            splitimage = np.where(conn_idx, img, BACKGROUND)
            #             minx=0
            #             miny=0
            #             maxx=0
            #             maxy=0
            #             for m in range(h):
            #                 if sum(splitimage[m,:])!=0:
            #                     miny=m
            #                     break
            #             for n in range(w):
            #                 if sum(splitimage[:,n])!=0:
            #                     minx=n
            #                     break
            #             for m in range(h-1,-1,-1):
            #                 if sum(splitimage[m,:])!=0:
            #                     maxy=m
            #                     break
            #             for n in range(w-1,-1,-1):
            #                 if sum(splitimage[:,n])!=0:
            #                     maxx=n
            #                     break

            (minx, maxx, miny, maxy) = _get_bound(splitimage)
            split_object = (splitimage[miny:maxy + 1, minx:maxx + 1]).tolist()

            regions.append({'start': (miny, minx), 'obj': split_object})
    return regions


def split_object(img):
    return _split_object(img, neighbor=4)


def split_object8(img):
    return _split_object(img, neighbor=8)


def split_object01(img):
    return _split_object01(img, neighbor=4)


def split_object801(img):
    return _split_object01(img, neighbor=8)


def h_one_color(A):
    b = []
    A = np.array(A)
    for i in range(len(A[0])):
        if len(np.unique(A[:, i])) == 1:
            b.append(i)
    return b


def w_one_color(A):
    b = []
    A = np.array(A)
    for i in range(len(A)):
        if len(np.unique(A[i, :])) == 1:
            b.append(i)
    return b


def maxcolor_b(A):
    A = np.array(A)
    b = np.bincount(A.flatten(), minlength=10)

    c = np.argsort(b)[-1]
    return c


# no count black
def Glue_1x1(List, m, n):
    ans = np.zeros((m, n), dtype=int)
    count = 0
    for i in range(m):
        for j in range(n):
            ans[i][j] = List[count]
            count += 1
    return ans.tolist()


# basic
def findcrossmap0(a0, b0):
    a = np.array(a0)
    b = np.array(b0)
    name_dic = {}
    crossdict = {}
    for i in range(len(a0) - 3 + 1):
        for j in range(len(a0[0]) - 3 + 1):
            if a0[i][j] == BACKGROUND and a0[i + 1][j] != BACKGROUND and a0[i + 2][j] == BACKGROUND and a0[i][
                j + 1] != BACKGROUND \
                    and a0[i + 1][j + 1] != BACKGROUND and a0[i + 2][j + 1] != BACKGROUND and a0[i][
                j + 2] == BACKGROUND and a0[i + 1][j + 2] != BACKGROUND \
                    and a0[i + 2][j + 2] == BACKGROUND:
                if str(a[i:i + 3, j:j + 3]) in crossdict.keys() and \
                        (crossdict[str(a[i:i + 3, j:j + 3])] != b[i:i + 3, j:j + 3]).any():
                    return -1
                else:
                    name_dic[str(a[i:i + 3, j:j + 3])] = a[i:i + 3, j:j + 3]
                    crossdict[str(a[i:i + 3, j:j + 3])] = b[i:i + 3, j:j + 3]
    if len(name_dic) == 0:
        return -1
    return name_dic, crossdict


def findcrossmap(basic_task):
    BACKGROUND = 0  #
    Input = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    Test_Case = Input[-1]
    Input = Input[:-1]
    name_dic = {}
    crossdict = {}
    for x, y in zip(Input, Output):
        x_array = np.array(x)
        y_array = np.array(y)
        if x_array.shape != y_array.shape:
            return -1
        for i in range(len(x) - 3 + 1):
            for j in range(len(x[0]) - 3 + 1):
                if x[i][j] == BACKGROUND and x[i + 1][j] != BACKGROUND and x[i + 2][j] == BACKGROUND and x[i][
                    j + 1] != BACKGROUND \
                        and x[i + 1][j + 1] != BACKGROUND and x[i + 2][j + 1] != BACKGROUND and x[i][
                    j + 2] == BACKGROUND and x[i + 1][j + 2] != BACKGROUND \
                        and x[i + 2][j + 2] == BACKGROUND:
                    if str(x_array[i:i + 3, j:j + 3]) in crossdict.keys() and \
                            (crossdict[str(x_array[i:i + 3, j:j + 3])] != y_array[i:i + 3, j:j + 3]).any():
                        return -1
                    else:
                        name_dic[str(x_array[i:i + 3, j:j + 3])] = x_array[i:i + 3, j:j + 3]
                        crossdict[str(x_array[i:i + 3, j:j + 3])] = y_array[i:i + 3, j:j + 3]
    if len(name_dic) == 0:
        return -1
    return name_dic, crossdict


def solve_cross_map(basic_task):
    Input = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    Test_Case = Input[-1]
    Input = Input[:-1]

    if findcrossmap(basic_task) == -1:
        return -1
    name_dic, crossdict = findcrossmap(basic_task)

    for x, y in zip(Input, Output):
        x_array = np.array(x)
        y_array = np.array(y)
        x_copy = x_array.copy()
        for i in range(len(x) - 3 + 1):
            for j in range(len(x[0]) - 3 + 1):
                if str(x_array[i:i + 3, j:j + 3]) in crossdict.keys():
                    x_copy[i:i + 3, j:j + 3] = crossdict[str(x_array[i:i + 3, j:j + 3])]
        if (x_copy != y_array).any():
            return -1
    Test_Case_array = np.array(Test_Case)
    Test_Case_copy = Test_Case_array.copy()
    for i in range(len(Test_Case) - 3 + 1):
        for j in range(len(Test_Case[0]) - 3 + 1):
            if str(Test_Case_array[i:i + 3, j:j + 3]) in crossdict.keys():
                Test_Case_copy[i:i + 3, j:j + 3] = crossdict[str(x_array[i:i + 3, j:j + 3])]
            else:
                for value in name_dic.values():
                    if checkColorMap(Test_Case_array[i:i + 3, j:j + 3], value) == True:
                        colormap = findColorMap(value, Test_Case_array[i:i + 3, j:j + 3])
                        Test_Case_copy[i:i + 3, j:j + 3] = np.array(applyColorMap(crossdict[str(value)], colormap))
                        break

    return Test_Case_copy.tolist()


# need count not black
def findcrossmap_line(basic_task):
    BACKGROUND = 0  #
    Input = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    Test_Case = Input[-1]
    Input = Input[:-1]
    name_dic = {}
    crossdict = {}
    for x, y in zip(Input, Output):
        x_array = np.array(x)
        y_array = np.array(y)
        for i in range(len(x)):
            for j in range(len(x[0])):
                if x[i][j] != 0:
                    c = x[i][j]
                    x_array[i, :] = c
                    x_array[:, j] = c
        x = x_array.tolist()

        if x_array.shape != y_array.shape:
            return -1
        for i in range(len(x) - 3 + 1):
            for j in range(len(x[0]) - 3 + 1):
                if x[i][j] == BACKGROUND and x[i + 1][j] != BACKGROUND and x[i + 2][j] == BACKGROUND and x[i][
                    j + 1] != BACKGROUND \
                        and x[i + 1][j + 1] != BACKGROUND and x[i + 2][j + 1] != BACKGROUND and x[i][
                    j + 2] == BACKGROUND and x[i + 1][j + 2] != BACKGROUND \
                        and x[i + 2][j + 2] == BACKGROUND:
                    if str(x_array[i:i + 3, j:j + 3]) in crossdict.keys() and \
                            (crossdict[str(x_array[i:i + 3, j:j + 3])] != y_array[i:i + 3, j:j + 3]).any():
                        return -1
                    else:
                        name_dic[str(x_array[i:i + 3, j:j + 3])] = x_array[i:i + 3, j:j + 3]
                        crossdict[str(x_array[i:i + 3, j:j + 3])] = y_array[i:i + 3, j:j + 3]
    if len(name_dic) == 0:
        return -1
    return name_dic, crossdict


def solve_cross_map_line(basic_task):
    Input = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    Test_Case = Input[-1]
    Input = Input[:-1]
    if findcrossmap_line(basic_task) == -1:
        return -1
    name_dic, crossdict = findcrossmap_line(basic_task)
    for x, y in zip(Input, Output):
        x_array = np.array(x)
        y_array = np.array(y)
        # if count>10:
        for i in range(len(x)):
            for j in range(len(x[0])):
                if x[i][j] != 0:
                    c = x[i][j]
                    x_array[i, :] = c
                    x_array[:, j] = c
        x_copy = x_array.copy()
        for i in range(len(x) - 3 + 1):
            for j in range(len(x[0]) - 3 + 1):
                if str(x_array[i:i + 3, j:j + 3]) in crossdict.keys():
                    x_copy[i:i + 3, j:j + 3] = crossdict[str(x_array[i:i + 3, j:j + 3])]
        if (x_copy != y_array).any():
            return -1

    Test_Case_array = np.array(Test_Case)
    for i in range(len(Test_Case)):
        for j in range(len(Test_Case[0])):
            if Test_Case[i][j] != 0:
                c = Test_Case[i][j]
                Test_Case_array[i, :] = c
                Test_Case_array[:, j] = c
    Test_Case_copy = Test_Case_array.copy()
    for i in range(len(Test_Case) - 3 + 1):
        for j in range(len(Test_Case[0]) - 3 + 1):
            if str(Test_Case_array[i:i + 3, j:j + 3]) in crossdict.keys():
                Test_Case_copy[i:i + 3, j:j + 3] = crossdict[str(Test_Case_array[i:i + 3, j:j + 3])]
            else:
                for value in name_dic.values():
                    if checkColorMap(Test_Case_array[i:i + 3, j:j + 3], value) == True:
                        colormap = findColorMap(value, Test_Case_array[i:i + 3, j:j + 3])
                        Test_Case_copy[i:i + 3, j:j + 3] = np.array(applyColorMap(crossdict[str(value)], colormap))
                        break
    return Test_Case_copy.tolist()


def Match_color_negative(basic_task):
    # returns -1 if no match is found
    # returns  Transformed_Test_Case  if the mathching rule is found
    Input = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    Input = [take_negative(a) for a in basic_task[0]]
    if -1 in Input:
        return -1
    Test_Case = Input[-1]
    Input = Input[:-1]
    for i in range(len(color_select)):
        S = color_select[i]
        solved = True

        for x, y in zip(Input, Output):
            c = colorbycolor_select(x, S)
            x1 = np.array(x)
            transformed_x = np.bincount(x1.flatten(), minlength=10)[c]
            if transformed_x == 0 or transformed_x != subarray_count(y, x):
                # 可擴充
                solved = False
                break
        if solved == True:
            Test_Case1 = np.array(Test_Case)
            Transformed_Test_Case = np.bincount(Test_Case1.flatten(), minlength=10)[colorbycolor_select(Test_Case, S)]
            return colorbycolor_select(Test_Case, S), Transformed_Test_Case

    return -1


def Solve_mul_color_negative(basic_task):
    # returns -1 if no match is found
    # returns Transformed_Test_Case  if the mathching rule is found
    # for this notebook we only look at mul color match
    Input = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    Input = [take_negative(a) for a in Input]
    if -1 in Input:
        return -1
    Test_Case = Input[-1]
    Input = Input[:-1]
    same_ratio = True
    R_x = []
    R_y = []
    for x, y in zip(Input, Output):
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

    if same_ratio and min(R_x) == max(R_x) and min(R_y) == max(R_y) and min(R_x) == len(Test_Case) and min(R_y) == len(
            Test_Case[0]):
        r1 = min(R_y)
        r2 = min(R_x)
        if Match_color_negative(basic_task) == -1:
            return -1
        color, count = Match_color_negative(basic_task)

        Test_Case_list = []
        Test_Case_Wight_2 = []
        Test_Case_flatten = ((np.array(Test_Case)).flatten()).tolist()
        for panel in range(r1 * r2):
            Test_Case_list.append(Test_Case)

            if Test_Case_flatten[panel] != color:
                Test_Case_Wight_2.append(0)
            else:
                Test_Case_Wight_2.append(1)
        Partial_Solutions = []
        for panel in range(r1 * r2):
            Partial_Solutions.append((np.array(Test_Case_list[panel])) * Test_Case_Wight_2[panel])
        Transformed_Test_Case = Glue(Partial_Solutions, r1, r2)
        return Transformed_Test_Case

    return -1


# 需要整合
def Match_color_bound_negative(basic_task):
    # returns -1 if no match is found
    # returns  Transformed_Test_Case  if the mathching rule is found
    Input = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    Input = [get_bound_image(a) for a in basic_task[0]]
    Input = [take_negative(a) for a in basic_task[0]]
    if -1 in Input:
        return -1
    Test_Case = Input[-1]  #####

    Input = Input[:-1]
    for i in range(len(color_select)):
        S = color_select[i]
        solved = True

        for x, y in zip(Input, Output):

            c = colorbycolor_select(x, S)
            x1 = np.array(x)
            transformed_x = np.bincount(x1.flatten(), minlength=10)[c]
            if transformed_x == 0 or transformed_x != subarray_count(y, x):
                # 可擴充
                solved = False
                break
        if solved == True:
            Test_Case1 = np.array(Test_Case)
            Transformed_Test_Case = np.bincount(Test_Case1.flatten(), minlength=10)[colorbycolor_select(Test_Case, S)]
            return colorbycolor_select(Test_Case, S), Transformed_Test_Case

    return -1


def Solve_mul_color_bound_negative(basic_task):
    # returns -1 if no match is found
    # returns Transformed_Test_Case  if the mathching rule is found
    # for this notebook we only look at mul color match
    Input = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]

    Input = [get_bound_image(a) for a in basic_task[0]]

    Input = [take_negative(a) for a in Input]

    if -1 in Input:
        return -1
    Test_Case = Input[-1]

    #     Test_Case=get_bound_image(Test_Case0)    ########################
    Input = Input[:-1]
    same_ratio = True
    R_x = []
    R_y = []
    for x, y0 in zip(Input, Output):  ########
        #         x=get_bound_image(x0) ########
        #         if x == []:return -1 #######
        y = y0
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

    if same_ratio and min(R_x) == max(R_x) and min(R_y) == max(R_y) and min(R_x) == len(Test_Case) and min(R_y) == len(
            Test_Case[0]):
        r1 = min(R_y)
        r2 = min(R_x)

        if Match_color_bound_negative(basic_task) == -1:  ######
            return -1
        color, count = Match_color_bound_negative(basic_task)  ############

        Test_Case_list = []
        Test_Case_Wight_2 = []
        Test_Case_flatten = ((np.array(Test_Case)).flatten()).tolist()
        for panel in range(r1 * r2):
            Test_Case_list.append(Test_Case)

            if Test_Case_flatten[panel] != color:
                Test_Case_Wight_2.append(0)
            else:
                Test_Case_Wight_2.append(1)
        Partial_Solutions = []
        for panel in range(r1 * r2):
            Partial_Solutions.append((np.array(Test_Case_list[panel])) * Test_Case_Wight_2[panel])
        Transformed_Test_Case = Glue(Partial_Solutions, r1, r2)
        return Transformed_Test_Case

    return -1


def Solve_juxt(basic_task):
    # returns -1 if no match is found
    # returns Transformed_Test_Case  if the mathching rule is found
    # for this notebook we only look at mul color match
    Input = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    Test_Case = Input[-1]
    Input = Input[:-1]
    solved = True
    for x, y in zip(Input, Output):
        # plot_picture(x)
        pattern = []
        a = split_object801(x)
        for j in range(len(a)):

            if len(pattern) == 0:
                pattern.append(a[j]["obj"])
            else:
                add_key = True
                for k in range(len(pattern)):
                    if is_juxt(pattern[k], a[j]["obj"]) != 0 or issubarray(pattern[k], a[j]["obj"]) == True \
                            or (len(a[j]["obj"]) == 1 and len(a[j]["obj"][0]) == 1):
                        add_key = False
                        break

                    elif is_juxt(a[j]["obj"], pattern[k]) != 0:
                        pattern[k] = a[j]["obj"]
                        add_key = False
                        break
                if add_key == True:
                    pattern.append(a[j]["obj"])

        if len(pattern) < 1:
            return -1
        y_juxt = juxt_pattern_to_image0(pattern[0], x)
        for m in range(1, len(pattern)):
            y_juxt = juxt_pattern_to_image0(pattern[m], y_juxt)
        # plot_picture(y_juxt)
        if y != y_juxt:
            solved = False
            break
    if solved == True:
        x = Test_Case
        pattern = []
        a = split_object801(x)
        for j in range(len(a)):
            if len(pattern) == 0:
                pattern.append(a[j]["obj"])
            else:
                add_key = True
                for k in range(len(pattern)):
                    if is_juxt(pattern[k], a[j]["obj"]) != 0 or issubarray(pattern[k], a[j]["obj"]) == True \
                            or (len(a[j]["obj"]) == 1 and len(a[j]["obj"][0]) == 1):
                        add_key = False
                        break

                    elif is_juxt(a[j]["obj"], pattern[k]) != 0:
                        pattern[k] = a[j]["obj"]
                        add_key = False
                        break
                if add_key == True:
                    pattern.append(a[j]["obj"])

        if len(pattern) < 1:
            return -1
        y_juxt = juxt_pattern_to_image0(pattern[0], x)
        for m in range(1, len(pattern)):
            y_juxt = juxt_pattern_to_image0(pattern[m], y_juxt)
        return y_juxt

    else:
        return -1


def split_shape_match(basic_task):
    Input = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    Test_Case = Input[-1]
    Input = Input[:-1]
    res = True
    for x0, y0 in zip(Input, Output):
        y1, y2 = len(y0), len(y0[0])
        if len(h_one_color(x0)) == 0 or len(w_one_color(x0)) == 0:
            res = False
            break
        elif len(h_one_color(x0)) + 1 == y2 and len(w_one_color(x0)) + 1 == y1 and len(h_one_color(x0)) + 1 != len(
                x0[0]) \
                and len(w_one_color(x0)) + 1 != len(x0):
            pass
        else:
            res = False
            break
    return res


def Solve_split_shape_negative(basic_task):
    Input = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    Test_Case = Input[-1]
    Input = Input[:-1]
    colormaps = {}
    colorchage = True
    if split_shape_match(basic_task) == False:
        return -1
    for i in range(len(Geometric)):

        S = Geometric[i]
        solved = True
        for x0, y0 in zip(Input, Output):
            x0 = take_negative(x0)

            if x0 == -1:
                return -1
            x = np.array(x0)
            y = np.array(y0)
            split_image = []
            split_color = []
            wlist = w_one_color(x0)
            hlist = h_one_color(x0)
            wlist.insert(0, -1)
            wlist.append(len(x0))
            hlist.insert(0, -1)
            hlist.append(len(x0[0]))
            for i in range(len(wlist) - 1):
                for j in range(len(hlist) - 1):
                    split_image.append(x[wlist[i] + 1:wlist[i + 1], hlist[j] + 1:hlist[j + 1]])
            for image in split_image:
                split_color.append(maxcolor_b(image))
            res = Glue_1x1(split_color, len(w_one_color(x0)) + 1, len(h_one_color(x0)) + 1)
            transformed_x = Apply_geometric(S, res)

            if checkColorMap(transformed_x, y0) == False:
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

        #####predict
        if solved == True:
            if take_negative(Test_Case) == -1:
                return -1
            else:
                Test_Case = take_negative(Test_Case)
            Test_Case_array = np.array(Test_Case)
            split_image = []
            split_color = []

            wlist = w_one_color(Test_Case)
            hlist = h_one_color(Test_Case)
            wlist.insert(0, -1)
            wlist.append(len(Test_Case))
            hlist.insert(0, -1)
            hlist.append(len(Test_Case[0]))
            for i in range(len(wlist) - 1):
                for j in range(len(hlist) - 1):
                    split_image.append(Test_Case_array[wlist[i] + 1:wlist[i + 1], hlist[j] + 1:hlist[j + 1]])

            for image in split_image:
                split_color.append(maxcolor_b(image))
            res = Glue_1x1(split_color, len(w_one_color(Test_Case)) + 1, len(h_one_color(Test_Case)) + 1)
            if colorchage == True:
                Transformed_Test_Case = applyColorMap(Apply_geometric(S, res), colormaps)
            else:
                Transformed_Test_Case = Apply_geometric(S, res)

            return Transformed_Test_Case
    return -1


def Solve_split_shape(basic_task):
    Input = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    Test_Case = Input[-1]
    Input = Input[:-1]
    colormaps = {}
    colorchage = True
    if split_shape_match(basic_task) == False:
        return -1
    for i in range(len(Geometric)):

        S = Geometric[i]
        solved = True
        for x0, y0 in zip(Input, Output):
            x = np.array(x0)
            y = np.array(y0)
            split_image = []
            split_color = []
            wlist = w_one_color(x0)
            hlist = h_one_color(x0)
            wlist.insert(0, -1)
            wlist.append(len(x0))
            hlist.insert(0, -1)
            hlist.append(len(x0[0]))
            for i in range(len(wlist) - 1):
                for j in range(len(hlist) - 1):
                    split_image.append(x[wlist[i] + 1:wlist[i + 1], hlist[j] + 1:hlist[j + 1]])
            for image in split_image:
                split_color.append(maxcolor_b(image))
            res = Glue_1x1(split_color, len(w_one_color(x0)) + 1, len(h_one_color(x0)) + 1)
            transformed_x = Apply_geometric(S, res)

            if checkColorMap(transformed_x, y0) == False:
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

        #####predict
        if solved == True:
            Test_Case_array = np.array(Test_Case)
            split_image = []
            split_color = []

            wlist = w_one_color(Test_Case)
            hlist = h_one_color(Test_Case)
            wlist.insert(0, -1)
            wlist.append(len(Test_Case))
            hlist.insert(0, -1)
            hlist.append(len(Test_Case[0]))
            for i in range(len(wlist) - 1):
                for j in range(len(hlist) - 1):
                    split_image.append(Test_Case_array[wlist[i] + 1:wlist[i + 1], hlist[j] + 1:hlist[j + 1]])

            for image in split_image:
                split_color.append(maxcolor_b(image))
            res = Glue_1x1(split_color, len(w_one_color(Test_Case)) + 1, len(h_one_color(Test_Case)) + 1)
            if colorchage == True:
                Transformed_Test_Case = applyColorMap(Apply_geometric(S, res), colormaps)
            else:
                Transformed_Test_Case = Apply_geometric(S, res)

            return Transformed_Test_Case
    return -1
