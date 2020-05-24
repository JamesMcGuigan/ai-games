#!/usr/bin/env python3

##### 
##### ./submission/kaggle_compile.py src_james/ensemble/sample_sub/sample_sub_combine.py
##### 
##### 2020-05-24 02:50:55+01:00
##### 
##### origin	git@github.com:seshurajup/kaggle-arc.git (fetch)
##### origin	git@github.com:seshurajup/kaggle-arc.git (push)
##### 
#####   james-wip c81cf89 Solvers | work in progress - broken
##### * master    7792251 Ensemble | import functions from kaggle notebook
##### 
##### 7792251fffe4e87da4c583afa466e3337c92b1d0
##### 

#####
##### START src_james/settings.py
#####

# DOCS: https://www.kaggle.com/WinningModelDocumentationGuidelines
import os
import pathlib
try:    root_dir = pathlib.Path(__file__).parent.parent.absolute()
except: root_dir = ''

settings = {
    'verbose': True,
    'debug':   not os.environ.get('KAGGLE_KERNEL_RUN_TYPE'),
}

if os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
    settings['dir'] = {
        "data":        "../input/abstraction-and-reasoning-challenge/",
        "output":      "./",
    }
else:
    settings['dir'] = {
        "data":        os.path.join(root_dir, "./input"),
        "output":      os.path.join(root_dir, "./submission"),
    }

####################
if __name__ == '__main__':
    for dirname in settings['dir'].values():
        try:    os.makedirs(dirname, exist_ok=True)  # BUGFIX: read-only filesystem
        except: pass
    # for key,value in settings.items():  print(f"settings['{key}']:".ljust(30), str(value))


#####
##### END   src_james/settings.py
#####

#####
##### START src_james/ensemble/sample_sub/path.py
#####

import os
from pathlib import Path

# from src_james.settings import settings

mode     = 'test' if os.environ.get('KAGGLE_KERNEL_RUN_TYPE','') else 'test'
mode_dir = mode
if   mode=='eval':  mode_dir = 'evaluation'
elif mode=='train': mode_dir = 'training'
elif mode=='test':  mode_dir = 'test'
else: raise Exception(f'invalid mode: {mode}')

data_path       = Path(settings['dir']['data'])
task_path       = data_path / mode
training_path   = data_path / 'training'
evaluation_path = data_path / 'evaluation'
test_path       = data_path / 'test'
output_dir      = Path( settings['dir']['output'] )

#####
##### END   src_james/ensemble/sample_sub/path.py
#####

#####
##### START src_james/ensemble/sample_sub/example_grid.py
#####

example_grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]


#####
##### END   src_james/ensemble/sample_sub/example_grid.py
#####

#####
##### START src_james/ensemble/solvers/Solve_split_shape.py
#####

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

# from src_james.ensemble.sample_sub.path import data_path, training_path, evaluation_path

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


#####
##### END   src_james/ensemble/solvers/Solve_split_shape.py
#####

#####
##### START src_james/ensemble/util.py
#####

import numpy as np


def Defensive_Copy(A):
    n = len(A)
    k = len(A[0])
    L = np.zeros((n, k), dtype=np.int8)
    for i in range(n):
        for j in range(k):
            L[i, j] = 0 + A[i][j]
    return L.tolist()


def Create(task, task_id=0):
    n = len(task['train'])
    Input  = [Defensive_Copy(task['train'][i]['input'])  for i in range(n)]
    Output = [Defensive_Copy(task['train'][i]['output']) for i in range(n)]
    Input.append(Defensive_Copy(task['test'][task_id]['input']))
    return Input, Output


def flattener(pred):
    if pred is None: return ''
    pred = np.array(pred).astype(np.int8).tolist()
    str_pred = str([row for row in pred])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred


#####
##### END   src_james/ensemble/util.py
#####

#####
##### START src_james/ensemble/color_classes.py
#####

import numpy as np


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

    else:
        return -1


#####
##### END   src_james/ensemble/color_classes.py
#####

#####
##### START src_james/ensemble/colors.py
#####

import numpy as np

# from src_james.ensemble.util import Defensive_Copy


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


#####
##### END   src_james/ensemble/colors.py
#####

#####
##### START src_james/ensemble/transformations.py
#####

import numpy as np

# Transformations
# from src_james.ensemble.util import Defensive_Copy


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


#####
##### END   src_james/ensemble/transformations.py
#####

#####
##### START src_james/ensemble/split_object.py
#####

BACKGROUND = 0
import numpy as np

def _get_bound(img0):
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
    x0, x1, y0, y1 = _get_bound(img0)
    img = np.array(img0)
    return img[y0:y1 + 1, x0:x1 + 1].tolist()


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


def split_color(img0):
    img = np.array(img0)
    color = np.unique(img)
    return [np.where(img == c, c, 0) for c in color if c != BACKGROUND]


def split_color_crop(img0):
    color_image = []
    img = np.array(img0)
    color = np.unique(img)
    for c in color:
        if c != BACKGROUND:
            imgc = np.where(img == c, c, 0)
            x0, x1, y0, y1 = _get_bound(imgc)
            imgc = (imgc[y0:y1 + 1, x0:x1 + 1]).tolist()
            color_image.append(imgc)
    return color_image


def split_object(img):
    return _split_object(img, neighbor=4)


def split_object8(img):
    return _split_object(img, neighbor=8)


def split_object01(img):
    return _split_object01(img, neighbor=4)


def split_object801(img):
    return _split_object01(img, neighbor=8)


#####
##### END   src_james/ensemble/split_object.py
#####

#####
##### START src_james/ensemble/resize.py
#####

import numpy as np

#array
def resize_o(a,r1,r2):
    try:
        return np.repeat(np.repeat(a, r1, axis=0), r2, axis=1)
    except:
        return a

def resize_c(a):
    c = np.count_nonzero(np.bincount(a.flatten(),minlength=10)[1:])
    return np.repeat(np.repeat(a, c, axis=0), c, axis=1)

def resize_2(a):
    return np.repeat(np.repeat(a, 2, axis=0), 2, axis=1)

def resize_3(a):
    return np.repeat(np.repeat(a, 2, axis=0), 2, axis=1)

resize_type=[resize_o,resize_c]

#####
##### END   src_james/ensemble/resize.py
#####

#####
##### START src_james/ensemble/period.py
#####

import numpy as np

# from src_james.ensemble.util import Defensive_Copy


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


#####
##### END   src_james/ensemble/period.py
#####

#####
##### START src_james/ensemble/mode.py
#####

# from src_james.ensemble.colors import object_dict, object_01area_dict, object_color_dict


def max_mode(A1):
    stats, name_dic = object_dict(A1)
    max_key = max(stats, key=lambda k: stats[k])
    return max_key, name_dic


def min_mode(A1):
    stats, name_dic = object_dict(A1)
    min_key = min(stats, key=lambda k: stats[k])
    return min_key, name_dic


# object_01area_dict
def max_01_area(A1):
    stats, name_dic = object_01area_dict(A1)
    max_key = max(stats, key=lambda k: stats[k])
    return max_key, name_dic


def min_01_area(A1):
    stats, name_dic = object_01area_dict(A1)
    min_key = min(stats, key=lambda k: stats[k])
    return min_key, name_dic


mode_select = [max_mode, min_mode, max_01_area, min_01_area]


def max_color_mode(A1, c):
    if object_color_dict(A1, c):
        stats, name_dic = object_color_dict(A1, c)
        max_key = max(stats, key=lambda k: stats[k])
        return max_key, name_dic
    else:
        return False


def min_color_mode(A1, c):
    if object_color_dict(A1, c):
        stats, name_dic = object_color_dict(A1, c)
        min_key = min(stats, key=lambda k: stats[k])
        return min_key, name_dic
    else:
        return False
   

#####
##### END   src_james/ensemble/mode.py
#####

#####
##### START src_james/ensemble/subarray.py
#####

import numpy as np

# A0 > B0
# from src_james.ensemble.util import Defensive_Copy


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


def issubarray_match(basic_task):
    Input  = [ Defensive_Copy(x) for x in basic_task[0] ]
    Output = [ Defensive_Copy(y) for y in basic_task[1] ]
    Test_Case = Input[-1]
    Input = Input[:-1]
    res = True
    for i in range(len(Input)):
        A = Input[i]
        B = Output[i]
        if issubarray(A, B) == False:
            res = False
            break
    return res


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


#####
##### END   src_james/ensemble/subarray.py
#####

#####
##### START src_james/ensemble/split_direction.py
#####

import numpy as np

# task=Trains[j]["train"]
def has_vertical_split0(task):
    same_mid_color = set()
    for i in range(len(task)):
        a = task[i]["input"]
        a = np.array(a)
        b = task[i]["output"]
        b = np.array(b)

        if a.shape[1] % 2 == 0:
            return False
        mid = int(np.floor(a.shape[1] / 2))
        mid_color = np.unique(a[:, mid])
        if mid_color.shape[0] > 1:
            return False
        mid_color = mid_color[0]
        same_mid_color.add(mid_color)
        if len(same_mid_color) > 1:
            return False
        if b.shape[1] != mid or b.shape[0] != a.shape[0]:
            return False
    return True


# task=Trains[j]["train"]
def has_horizontal_split0(task):
    same_mid_color = set()
    for i in range(len(task)):
        a = task[i]["input"]
        a = np.array(a)
        b = task[i]["output"]
        b = np.array(b)

        if a.shape[0] % 2 == 0:
            return False
        mid = int(np.floor(a.shape[0] / 2))
        mid_color = np.unique(a[mid, :])
        if mid_color.shape[0] > 1:
            return False
        mid_color = mid_color[0]
        same_mid_color.add(mid_color)
        if len(same_mid_color) > 1:
            return False
        if b.shape[0] != mid or b.shape[1] != a.shape[1]:
            return False
    return True


def vertical_split0(a, b):
    a = np.array(a)
    b = np.array(b)
    if a.shape[1] % 2 != 0 or a.shape[1] < 2:
        return False
    if a.shape[1] != b.shape[1] * 2 or a.shape[0] != b.shape[0]:
        return False

    mid = int(np.floor(a.shape[1] / 2))
    return a[:, :mid].tolist(), a[:, mid:].tolist()


def horizontal_split0(a, b):
    a = np.array(a)
    b = np.array(b)

    if a.shape[0] % 2 != 0 or a.shape[0] < 2:
        return False
    if a.shape[0] != b.shape[0] * 2 or a.shape[1] != b.shape[1]:
        return False

    mid = int(np.floor(a.shape[0] / 2))

    return a[:mid, :].tolist(), a[mid:, :].tolist()


# task=Trains[j]["train"]
def vertical_split(a, b):
    a = np.array(a)
    b = np.array(b)
    if a.shape[1] % 2 == 0:
        return False
    mid = int(np.floor(a.shape[1] / 2))
    mid_color = np.unique(a[:, mid])
    if mid_color.shape[0] > 1:
        return False
    #     mid_color = mid_color[0]
    #     same_mid_color.add(mid_color)
    #     if len(same_mid_color)>1:
    #         return False
    if b.shape[1] != mid or b.shape[0] != a.shape[0]:
        return False
    return a[:, :mid].tolist(), a[:, mid + 1:].tolist()


# task=Trains[j]["train"]
def horizontal_split(a, b):
    a = np.array(a)
    b = np.array(b)

    if a.shape[0] % 2 == 0:
        return False
    mid = int(np.floor(a.shape[0] / 2))
    mid_color = np.unique(a[mid, :])
    if mid_color.shape[0] > 1:
        return False
    #     mid_color = mid_color[0]
    #     same_mid_color.add(mid_color)
    #     if len(same_mid_color)>1:
    #         return False
    if b.shape[0] != mid or b.shape[1] != a.shape[1]:
        return False
    return a[:mid, :].tolist(), a[mid + 1:, :].tolist()


#####
##### END   src_james/ensemble/split_direction.py
#####

#####
##### START src_james/ensemble/crop.py
#####

import numpy as np

def crop_min_nonb(a0):
    try:
        a = np.array(a0)
        b = np.bincount(a.flatten(), minlength=10)
        b1 = np.delete(b, 0)
        c = int(np.where(b == np.min(b1[np.nonzero(b1)]))[0])
        coords = np.argwhere(a == c)
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        return a[x_min:x_max + 1, y_min:y_max + 1].tolist()
    except:
        return a0.tolist()


def crop_min_nonb_1(a0):
    try:
        a = np.array(a0)
        b = np.bincount(a.flatten(), minlength=10)
        b1 = np.delete(b, 0)
        c = int(np.where(b == np.min(b1[np.nonzero(b1)]))[0])
        coords = np.argwhere(a == c)
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        return a[x_min - 1:x_max + 2, y_min - 1:y_max + 2].tolist()
    except:
        return a0.tolist()


def crop_max(a0):
    a = np.array(a0)
    try:
        b = np.bincount(a.flatten(), minlength=10)
        b[0] = 500  # ???255
        c = np.argsort(b)[-2]
        coords = np.argwhere(a == c)
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        return a[x_min:x_max + 1, y_min:y_max + 1].tolist()
    except:
        return a.tolist()


def crop_max_1(a0):
    a = np.array(a0)
    try:
        b = np.bincount(a.flatten(), minlength=10)
        b[0] = 500  # ???255
        c = np.argsort(b)[-2]
        coords = np.argwhere(a == c)
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        return a[x_min - 1:x_max + 2, y_min - 1:y_max + 2].tolist()
    except:
        return a.tolist()


def _get_bound_not(img0, c):
    img = np.array(img0)
    h, w = img.shape
    x0 = w - 1
    x1 = 0
    y0 = h - 1
    y1 = 0
    for x in range(w):
        for y in range(h):
            if img[y, x] != c:
                continue
            x0 = min(x0, x)
            x1 = max(x1, x)
            y0 = min(y0, y)
            y1 = max(y1, y)
    return x0, x1, y0, y1


def crop_c(img0, c):
    x0, x1, y0, y1 = _get_bound_not(img0, c)
    img = np.array(img0)
    if c not in img:
        return -1
    return img[y0:y1 + 1, x0:x1 + 1].tolist()


def crop_c_1(img0, c):
    x0, x1, y0, y1 = _get_bound_not(img0, c)
    img = np.array(img0)
    if c not in img:
        return -1
    try:
        return img[y0 + 1:y1, x0 + 1:x1].tolist()
    except:
        return -1


crop_mode   = [crop_min_nonb, crop_max, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
crop_mode_1 = [crop_min_nonb_1, crop_max_1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0]


# crop_mode=[crop_min_nonb,crop_max]
def Apply_crop_mode(mode, x):
    if type(mode) == int:
        if crop_c(x, mode) != -1:
            return crop_c(x, mode)
        else:

            return -1
    else:
        return mode(x)


def Apply_crop_mode_1(mode, x):
    if type(mode) == int:
        if crop_c_1(x, mode) != -1:
            return crop_c_1(x, mode)
        else:

            return -1
    else:
        return mode(x)


#####
##### END   src_james/ensemble/crop.py
#####

#####
##### START src_james/ensemble/solvers/Match_crop_mode.py
#####

import numpy as np

# from src_james.ensemble.crop import crop_mode, crop_mode_1, Apply_crop_mode
# from src_james.ensemble.util import Defensive_Copy


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


#####
##### END   src_james/ensemble/solvers/Match_crop_mode.py
#####

#####
##### START src_james/ensemble/solvers/Recolor.py
#####

import numpy as np

def Recolor(task):
    Input = task[0]
    Output = task[1]
    Test_Picture = Input[-1]
    Input = Input[:-1]
    N = len(Input)
    list_lenc = []

    for x, y in zip(Input, Output):
        if len(x) != len(y) or len(x[0]) != len(y[0]):
            return -1
        lenc = len(np.unique(np.array(y)))
        list_lenc.append(lenc)
    Best_Dict = -1
    Best_Q1 = -1
    Best_Q2 = -1
    Best_v = -1
    # v ranges from 0 to 3. This gives an extra flexibility of measuring distance from any of the 4 corners
    Pairs = []
    for t in range(15):
        for Q1 in range(1, 8):
            for Q2 in range(1, 8):
                if Q1 + Q2 == t:
                    Pairs.append((Q1, Q2))
    Pairs.append((15, 15))
    Pairs.append((30, 30))

    for Q1, Q2 in Pairs:
        for v in range(4):

            if Best_Dict != -1:
                continue
            possible = True
            Dict = {}

            for x, y in zip(Input, Output):
                n = len(x)
                k = len(x[0])
                for i in range(n):
                    for j in range(k):
                        if v == 0 or v == 2:
                            p1 = i % Q1
                        else:
                            p1 = (n - 1 - i) % Q1
                        if v == 0 or v == 3:
                            p2 = j % Q2
                        else:
                            p2 = (k - 1 - j) % Q2
                        color1 = x[i][j]
                        color2 = y[i][j]
                        if color1 != color2:
                            rule = (p1, p2, color1)

                            if rule not in Dict:
                                Dict[rule] = color2
                            elif Dict[rule] != color2:
                                possible = False
            # print(Dict)
            if possible:

                # Let's see if we actually solve the problem
                for x, y in zip(Input, Output):
                    n = len(x)
                    k = len(x[0])
                    for i in range(n):
                        for j in range(k):
                            if v == 0 or v == 2:
                                p1 = i % Q1
                            else:
                                p1 = (n - 1 - i) % Q1
                            if v == 0 or v == 3:
                                p2 = j % Q2
                            else:
                                p2 = (k - 1 - j) % Q2

                            color1 = x[i][j]
                            rule = (p1, p2, color1)

                            if rule in Dict:
                                color2 = 0 + Dict[rule]
                            else:
                                color2 = 0 + y[i][j]
                            if color2 != y[i][j]:
                                possible = False

                if possible:
                    Best_Dict = Dict
                    Best_Q1 = Q1
                    Best_Q2 = Q2
                    Best_v = v

    if Best_Dict == -1:
        return -1  # meaning that we didn't find a rule that works for the traning cases

    # Otherwise there is a rule: so let's use it:
    n = len(Test_Picture)
    k = len(Test_Picture[0])

    answer = np.zeros((n, k), dtype=int)

    for i in range(n):
        for j in range(k):
            if Best_v == 0 or Best_v == 2:
                p1 = i % Best_Q1
            else:
                p1 = (n - 1 - i) % Best_Q1
            if Best_v == 0 or Best_v == 3:
                p2 = j % Best_Q2
            else:
                p2 = (k - 1 - j) % Best_Q2

            color1 = Test_Picture[i][j]
            rule = (p1, p2, color1)
            if (p1, p2, color1) in Best_Dict:
                answer[i][j] = 0 + Best_Dict[rule]
            else:
                answer[i][j] = 0 + color1

            # print(answer)
    # print(list_lenc)
    if len(np.unique(list_lenc)) == 1:
        # print(answer)
        if len(np.unique(answer)) == list_lenc[0]:
            return answer.tolist()
        else:
            return -1

    return answer.tolist()


#####
##### END   src_james/ensemble/solvers/Recolor.py
#####

#####
##### START src_james/ensemble/solvers/Recolor0.py
#####

import numpy as np

# from src_james.ensemble.split_object import get_bound_image


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


#####
##### END   src_james/ensemble/solvers/Recolor0.py
#####

#####
##### START src_james/ensemble/solvers/Solve_color_crop.py
#####

import numpy as np

# from src_james.ensemble.colors import color_select, colorbycolor_select, cropbycolor, checkColorMap, findColorMap
# from src_james.ensemble.colors import applyColorMap, mergedict
# from src_james.ensemble.util import Defensive_Copy


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


#####
##### END   src_james/ensemble/solvers/Solve_color_crop.py
#####

#####
##### START src_james/ensemble/solvers/Solve_connect.py
#####

import numpy as np

# from src_james.ensemble.util import Defensive_Copy

BACKGROUND = 0
def connect_dot_row(a0):
    a = np.array(a0)
    a_copy = a.copy()
    m, n = a.shape
    for i in range(m):
        for j in range(n):
            if a[i][j] == BACKGROUND:
                continue
            else:
                c = a[i][j]
            if j + 1 <= n - 1:
                for k in range(j + 1, n):
                    if a[i][k] == c:
                        for p in range(j + 1, k):
                            if a_copy[i, p] == BACKGROUND:
                                a_copy[i, p] = c

            if i + 1 <= m - 1:
                for l in range(i + 1, m):
                    if a[l][j] == c:
                        a_copy[i + 1:l, j] = c
                        for q in range(i + 1, l):
                            if a_copy[q, j] == BACKGROUND:
                                a_copy[q, j] = c

    return a_copy.tolist()


def solve_connect(basic_task):
    Input  = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    Test_Case = Input[-1]
    Input = Input[:-1]

    for x, y in zip(Input, Output):
        pred_y = connect_dot_row(x)
        if pred_y != y:
            return -1
    return connect_dot_row(Test_Case)


#####
##### END   src_james/ensemble/solvers/Solve_connect.py
#####

#####
##### START src_james/ensemble/solvers/Solve_filing.py
#####

# from src_james.ensemble.colors import color_add_task, filling, checkColorMap, findColorMap, mergedict, applyColorMap
# from src_james.ensemble.util import Defensive_Copy


def Solve_filling(basic_task):
    Input = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    Test_Case = Input[-1]
    Input = Input[:-1]
    colormaps = {}
    if color_add_task(basic_task) == -1 or len(color_add_task(basic_task)) < 1:
        return -1
    else:
        c = color_add_task(basic_task)[0]
    for x, y in zip(Input, Output):
        pre_y = filling(x, c)
        if checkColorMap(pre_y, y) == False:
            return -1
        else:
            colormap = findColorMap(pre_y, y)
        if mergedict([colormaps, colormap]) == False:
            return -1
        colormaps = mergedict([colormaps, colormap])
        pre_y = applyColorMap(pre_y, colormaps)
        if pre_y != y:
            return -1

    res = applyColorMap(filling(Test_Case, c), colormaps)
    return res


#####
##### END   src_james/ensemble/solvers/Solve_filing.py
#####

#####
##### START src_james/ensemble/solvers/Solve_inoutmap.py
#####

# plit_object801
import numpy as np

# from src_james.ensemble.colors import checkColorMap, findColorMap, applyColorMap
# from src_james.ensemble.split_object import split_object801
# from src_james.ensemble.util import Defensive_Copy


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


#####
##### END   src_james/ensemble/solvers/Solve_inoutmap.py
#####

#####
##### START src_james/ensemble/solvers/Solve_logtic.py
#####

import numpy as np

# from src_james.ensemble.colors import checkColorMap, findColorMap, applyColorMap
# from src_james.ensemble.split_direction import vertical_split, horizontal_split, vertical_split0, horizontal_split0
# from src_james.ensemble.transformations import Vert, Hor
# from src_james.ensemble.util import Defensive_Copy


def logitic_And(a, b):
    a = np.array(a)
    b = np.array(b)
    c = np.zeros(a.shape)
    if a.shape != b.shape:
        return -1
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if a[i][j] != 0 and a[i][j] == b[i][j]:
                c[i][j] = a[i][j]
    return c


def logitic_And_01(a, b):
    a = np.array(a)
    b = np.array(b)
    a = np.where(a, 1, 0)
    b = np.where(b, 1, 0)

    c = np.zeros(a.shape)
    if a.shape != b.shape:
        return -1
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if a[i][j] != 0 and a[i][j] == b[i][j]:
                c[i][j] = a[i][j]
    return c


def logitic_Or(a, b):
    a = np.array(a)
    b = np.array(b)
    c = np.zeros(a.shape)
    if a.shape != b.shape:
        return -1
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if a[i][j] == 0 and b[i][j] != 0:
                c[i][j] = b[i][j]
            elif a[i][j] != 0 and b[i][j] == 0:
                c[i][j] = a[i][j]
            elif a[i][j] != 0 and a[i][j] == b[i][j]:
                c[i][j] = a[i][j]
    return c


def logitic_Or_01(a, b):
    a = np.array(a)
    b = np.array(b)
    a = np.where(a, 1, 0)
    b = np.where(b, 1, 0)
    c = np.zeros(a.shape)
    if a.shape != b.shape:
        return -1
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if a[i][j] == 0 and b[i][j] != 0:
                c[i][j] = b[i][j]
            elif a[i][j] != 0 and b[i][j] == 0:
                c[i][j] = a[i][j]
            elif a[i][j] != 0 and a[i][j] == b[i][j]:
                c[i][j] = a[i][j]
    return c


def logitic_Xor(a, b):
    a = np.array(a)
    b = np.array(b)
    c = np.zeros(a.shape)
    if a.shape != b.shape:
        return -1
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if a[i][j] == 0 and b[i][j] != 0:
                c[i][j] = b[i][j]
            elif a[i][j] != 0 and b[i][j] == 0:
                c[i][j] = a[i][j]
    return c


def logitic_Xor_01(a, b):
    a = np.array(a)
    b = np.array(b)
    a = np.where(a, 1, 0)
    b = np.where(b, 1, 0)
    c = np.zeros(a.shape)
    if a.shape != b.shape:
        return -1
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if a[i][j] == 0 and b[i][j] != 0:
                c[i][j] = b[i][j]
            elif a[i][j] != 0 and b[i][j] == 0:
                c[i][j] = a[i][j]
    return c


# need color change
def logitic_Nor(a, b):
    a = np.array(a)
    b = np.array(b)
    c = np.zeros(a.shape)
    if a.shape != b.shape:
        return -1
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if a[i][j] == 0 and b[i][j] == 0:
                c[i][j] = 1
    return c


logitic_map = [logitic_And, logitic_Or, logitic_Nor, logitic_Xor, logitic_And_01, logitic_Or_01, logitic_Xor_01]


def Apply_logitic(S, x, y):
    if S in logitic_map:
        x1 = Defensive_Copy(x)
        y1 = Defensive_Copy(y)
        z1 = S(x1, y1)
    return z1.tolist()


def Solve_logtic(basic_task):
    Input  = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    Test_Case = Input[-1]
    Input = Input[:-1]
    for i in range(len(logitic_map)):
        S = logitic_map[i]
        solved = True
        m = 0

        for x, y in zip(Input, Output):
            if vertical_split(x, y) or horizontal_split(x, y) or vertical_split0(x, y) or horizontal_split0(x, y):
                if horizontal_split(x, y):
                    a, b = horizontal_split(x, y)
                    b1 = Vert(b)
                    mask = 0

                elif horizontal_split0(x, y):
                    a, b = horizontal_split0(x, y)
                    b1 = Vert(b)
                    mask = 0
                elif vertical_split(x, y):
                    a, b = vertical_split(x, y)
                    b1 = Hor(b)
                    mask = 1

                else:
                    a, b = vertical_split0(x, y)
                    b1 = Hor(b)
                    mask = 1

            else:
                return -1
                # print(a,b)
            logitic_x = Apply_logitic(S, a, b)
            logitic_x1 = Apply_logitic(S, a, b1)
            # print(logitic_x1)
            # color chage

            if checkColorMap(logitic_x, y) and logitic_x != y:
                m = 1
                colormap  = findColorMap(logitic_x, y)
                logitic_x = applyColorMap(logitic_x, colormap)
            if checkColorMap(logitic_x1, y) and logitic_x1 != y:
                m = 1
                colormap = findColorMap(logitic_x1, y)
                logitic_x1 = applyColorMap(logitic_x1, colormap)

            if logitic_x != y and logitic_x1 != y:
                solved = False
                break

        if solved == True:

            if logitic_x == y:
                if mask == 0 and horizontal_split(Test_Case, y):
                    a, b = horizontal_split(Test_Case, y)
                elif mask == 0 and horizontal_split0(Test_Case, y):
                    a, b = horizontal_split0(Test_Case, y)

                elif mask == 1 and vertical_split(Test_Case, y):
                    a, b = vertical_split(Test_Case, y)

                elif mask == 1 and vertical_split0(Test_Case, y):
                    a, b = vertical_split0(Test_Case, y)
                # 187
                else:
                    return -1
                logitic_Test_Case = Apply_logitic(S, a, b)

                if m == 1:
                    logitic_Test_Case = applyColorMap(logitic_Test_Case, colormap)
                # to int
                logitic_Test_Case = [[int(s) for s in a] for a in logitic_Test_Case]

                return logitic_Test_Case
            else:
                if mask == 0 and horizontal_split(Test_Case, y):
                    a, b = horizontal_split(Test_Case, y)
                    b1 = Vert(b)
                elif mask == 0 and horizontal_split0(Test_Case, y):
                    a, b = horizontal_split0(Test_Case, y)
                    b1 = Vert(b)
                elif mask == 1 and vertical_split(Test_Case, y):
                    a, b = vertical_split(Test_Case, y)
                    b1 = Hor(b)

                elif mask == 1 and vertical_split0(Test_Case, y):
                    a, b = vertical_split0(Test_Case, y)
                    b1 = Hor(b)

                else:
                    return -1

                logitic_Test_Case = Apply_logitic(S, a, b1)

                if m == 1:
                    logitic_Test_Case = applyColorMap(logitic_Test_Case, colormap)
                # to int
                logitic_Test_Case = [[int(s) for s in a] for a in logitic_Test_Case]

                return logitic_Test_Case
    return -1


#####
##### END   src_james/ensemble/solvers/Solve_logtic.py
#####

#####
##### START src_james/ensemble/solvers/Solve_mode_dict.py
#####

import numpy as np

# from src_james.ensemble.util import Defensive_Copy


def findmodemap(A, B):
    A_array = np.array(A)
    B_pad = np.pad(B, ((1, 1), (1, 1)), "constant", constant_values=(-1, -1))
    A_pad = np.pad(A, ((1, 1), (1, 1)), "constant", constant_values=(-1, -1))
    m, n = A_pad.shape
    total_dict = {}
    A1 = A_pad.copy()
    A2 = A_pad.copy()
    for k in range(30):
        dict1 = {}
        for i in range(m):
            for j in range(n):

                if A1[i, j] != -1 and A1[i, j] != 0:
                    if str(A1[i - 1:i + 2, j - 1:j + 2]) not in dict1:
                        dict1[str(A1[i - 1:i + 2, j - 1:j + 2])] = B_pad[i - 1:i + 2, j - 1:j + 2]
                    if str(A1[i - 1:i + 2, j - 1:j + 2]) in dict1 and (
                            dict1[str(A1[i - 1:i + 2, j - 1:j + 2])] != B_pad[i - 1:i + 2, j - 1:j + 2]).any():
                        return -1

        total_dict = dict(dict1, **total_dict)
        # print(total_dict)
        A1_copy = A1.copy()
        # A1_copy=A1
        for i in range(m):
            for j in range(n):

                if str(A1_copy[i - 1:i + 2, j - 1:j + 2]) in total_dict.keys():
                    A2[i - 1:i + 2, j - 1:j + 2] = total_dict[str(A1_copy[i - 1:i + 2, j - 1:j + 2])]

        # plot_picture(A2.tolist())
        #         if (A1==B_pad).all():
        #             #print(k)
        #             break
        #         else:
        A1 = A2
        # plot_picture(A1.tolist())

    return total_dict


def usemodedict(A, total_dict):
    A_array = np.array(A)
    A_pad = np.pad(A, ((1, 1), (1, 1)), "constant", constant_values=(-1, -1))
    m, n = A_pad.shape
    total_dict = total_dict
    A1 = A_pad.copy()
    A2 = A_pad.copy()
    for k in range(100):
        A1_copy = A1.copy()
        for i in range(m):
            for j in range(n):
                if str(A1_copy[i - 1:i + 2, j - 1:j + 2]) in total_dict.keys():
                    # print(A1_copy[i-1:i+2,j-1:j+2])

                    A2[i - 1:i + 2, j - 1:j + 2] = total_dict[str(A1_copy[i - 1:i + 2, j - 1:j + 2])]

        A1 = A2
    return A2[1:-1, 1:-1].tolist()


def Solve_mode_dict(basic_task):
    Input  = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    Test_Case = Input[-1]
    Input = Input[:-1]
    total_dict = {}
    for i in range(10):
        mask = False
        for j in range(len(Input)):
            if i in Test_Case and i in Input[j]:
                mask = True
                break
            elif i not in Test_Case:
                mask = True
                break
        if mask == False:
            return -1

    for x, y in zip(Input, Output):
        if len(x) != len(y) or len(x[0]) != len(y[0]):
            return -1
        if findmodemap(x, y) == -1:
            return -1
        else:
            total_dict = dict(total_dict, **(findmodemap(x, y)))

    for x, y in zip(Input, Output):
        if y != usemodedict(x, total_dict):
            return -1

    return usemodedict(Test_Case, total_dict)



#####
##### END   src_james/ensemble/solvers/Solve_mode_dict.py
#####

#####
##### START src_james/ensemble/solvers/Solve_mul_color_bound.py
#####

import numpy as np

# 需要整合
# from src_james.ensemble.colors import color_select, colorbycolor_select
# from src_james.ensemble.split_object import get_bound_image
# from src_james.ensemble.subarray import subarray_count
# from src_james.ensemble.transformations import Glue
# from src_james.ensemble.util import Defensive_Copy


def Match_color_bound(basic_task):
    # returns -1 if no match is found
    # returns  Transformed_Test_Case  if the mathching rule is found
    Input = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    Test_Case0 = Input[-1]  #####
    Test_Case = get_bound_image(Test_Case0)  ####
    Input = Input[:-1]
    for i in range(len(color_select)):
        S = color_select[i]
        solved = True

        for x0, y in zip(Input, Output):
            x = get_bound_image(x0)  ######
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


def Solve_mul_color_bound(basic_task):
    # returns -1 if no match is found
    # returns Transformed_Test_Case  if the mathching rule is found
    # for this notebook we only look at mul color match
    Input  = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    Test_Case0 = Input[-1]
    Test_Case = get_bound_image(Test_Case0)  ########################
    Input = Input[:-1]
    same_ratio = True
    R_x = []
    R_y = []
    for x0, y0 in zip(Input, Output):  ########
        x = get_bound_image(x0)  ########
        if x == []: return -1  #######
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

        if Match_color_bound(basic_task) == -1:  ######
            return -1
        color, count = Match_color_bound(basic_task)  ############

        Test_Case_list    = []
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


#####
##### END   src_james/ensemble/solvers/Solve_mul_color_bound.py
#####

#####
##### START src_james/ensemble/solvers/Solve_negative.py
#####

import numpy as np

# from src_james.ensemble.color_classes import take_negative
# from src_james.ensemble.colors import checkColorMap, findColorMap, mergedict, applyColorMap
# from src_james.ensemble.split_object import get_bound_image
# from src_james.ensemble.util import Defensive_Copy


def Solve_negative(basic_task):
    # returns -1 if no match is found
    # returns  Transformed_Test_Case  if the matching rule is found
    Input  = [ Defensive_Copy(x) for x in basic_task[0] ]
    Output = [ Defensive_Copy(y) for y in basic_task[1] ]
    Test_Case = Input[-1]
    Input     = Input[:-1]
    colormaps = {}

    for x0, y in zip(Input, Output):
        x = np.array(x0)
        x = get_bound_image(x)
        if take_negative(x) != -1:
            negative_x = take_negative(x)
        else:
            return -1

        if checkColorMap(negative_x, y) == False:
            return -1
        else:
            colormap = findColorMap(negative_x, y)

        if mergedict([colormaps, colormap]) == False:
            return -1

        colormaps  = mergedict([colormaps, colormap])
        pre_y_list = applyColorMap(negative_x, colormaps)

        if pre_y_list != y:
            return -1

    Test_Case_pred = applyColorMap(take_negative(take_negative(Test_Case)), colormaps)
    return Test_Case_pred


#####
##### END   src_james/ensemble/solvers/Solve_negative.py
#####

#####
##### START src_james/ensemble/solvers/Solve_object_mode_color.py
#####

# from src_james.ensemble.mode import max_color_mode, min_color_mode, mode_select
# from src_james.ensemble.split_object import split_object801, split_object8, split_object01, split_object
# from src_james.ensemble.util import Defensive_Copy

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


#####
##### END   src_james/ensemble/solvers/Solve_object_mode_color.py
#####

#####
##### START src_james/ensemble/solvers/Solve_output_color_change_mode.py
#####

import numpy as np

# from src_james.ensemble.colors import checkColorMap, applyColorMap
# from src_james.ensemble.util import Defensive_Copy


def Match_output_color_change_mode(basic_task):
    # returns -1 if no match is found
    # returns  Transformed_Test_Case  if the mathching rule is found
    Input = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    Test_Case = Input[-1]
    Input = Input[:-1]
    A = np.array(Output[0])
    solved = True
    if np.array(Input[0]).shape != np.array(A).shape:
        return False
    for i in range(1, len(Output)):
        B = np.array(Output[i])
        if len(np.unique(B)) == 1:
            solved = False
            break
        if checkColorMap(A, B) == False:
            solved = False
            break
    return solved


def solve_output_color_change_mode(basic_task):
    # returns -1 if no match is found
    # returns  Transformed_Test_Case  if the mathching rule is found
    Input = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    Test_Case = Input[-1]
    Input = Input[:-1]
    if Match_output_color_change_mode(basic_task) == True:
        A = Output[0]
        use_c = -1
        for c in range(0, 10):
            res = True
            for x, y in zip(Input, Output):
                colormap = {}
                for i in range(len(x)):
                    for j in range(len(x[0])):
                        if x[i][j] != c and A[i][j] not in colormap:
                            colormap[A[i][j]] = x[i][j]

                B = applyColorMap(A, colormap)

                if B != y:
                    res = False
                    break

            if res == True:
                use_c = c
                break

        if use_c == -1:
            return -1
        else:
            colormap_T = {}
            for i in range(len(Test_Case)):
                for j in range(len(Test_Case[0])):
                    if Test_Case[i][j] != c and A[i][j] not in colormap_T:
                        colormap_T[A[i][j]] = Test_Case[i][j]

            B = applyColorMap(A, colormap_T)
            return B

    else:
        return -1
   

#####
##### END   src_james/ensemble/solvers/Solve_output_color_change_mode.py
#####

#####
##### START src_james/ensemble/solvers/Solve_patch.py
#####

import numpy as np
from skimage.util import view_as_windows


def getnoisecolor(Input, Output):
    noises = []
    for inp, oot in zip(Input[:-1], Output):
        to_unique = np.unique(inp)
        t1_unique = np.unique(oot)
        diff = np.setdiff1d(to_unique, t1_unique)
        if len(to_unique) <= len(t1_unique):
            return -1
        noises = np.concatenate((noises, diff), axis=0)
    #     if np.unique(noises).shape[0]==1:
    #         return np.unique(noises)[0]
    #     else:
    #         return -1
    noises = noises.astype(int)
    counts = np.bincount(noises)
    return np.argmax(counts)


from skimage.feature import greycomatrix


def getCoMat(X):
    cooc = greycomatrix(X, [1], [0], levels=10)
    return cooc[:, :, 0, 0]


def getBackground(comat):
    ratios = {}
    for idx, elm in enumerate(comat):
        if np.sum(elm) == 0:
            ratios[idx] = 0
            continue
        ratios[idx] = elm[idx] / np.sum(elm)
    ratios = sorted(ratios, key=ratios.get, reverse=True)
    return ratios


def Patch(t0):
    tp = np.pad(t0, 2, pad_with)
    x, y = t0.shape
    print(x, y)
    old_tp = np.zeros(tp.shape)
    while (not np.equal(old_tp, tp).all()):
        old_tp = np.copy(tp)
        for i in range(2, t0.shape[0] + 2):
            for j in range(2, t0.shape[1] + 2):
                probablities = np.zeros(10)
                if True:
                    for l in [-2, -1, 0, 1, 2]:
                        for m in [-2, -1, 0, 1, 2]:
                            if (tp[i + l, j + m] == 0):
                                continue
                            for c in range(10):
                                if (c == 0):
                                    continue
                                neighbor_color = tp[i + l, j + m]
                                joint_prob_indices = np.where(tp == c)
                                joint_prob_indices = [np.asarray(joint_prob_indices[0]),
                                                      np.asarray(joint_prob_indices[1])]
                                joint_prob_indices[0] += l
                                joint_prob_indices[1] += m
                                joint_prob_count = np.count_nonzero(tp[joint_prob_indices] == neighbor_color)
                                joint_prob = joint_prob_count
                                # joint2=np.where(tp==c)
                                # joint_2_count=0
                                # for id1,id2 in zip(joint2[0],joint2[1]):
                                #     if (tp[id1+l,id2+m]==neighbor_color):
                                #         joint_2_count+=1
                                margin_indices = np.where(tp != -1)
                                margin_indices = [np.asarray(margin_indices[0]), np.asarray(margin_indices[1])]
                                margin_indices[0] += l
                                margin_indices[1] += m
                                margin_count = np.count_nonzero(tp[margin_indices] == neighbor_color)
                                if (probablities[c] == 0):
                                    probablities[c] += (joint_prob / margin_count)
                                else:

                                    probablities[c] = max((joint_prob / margin_count), probablities[c])
                    # print(probablities)
                    tp[i, j] = np.argmax(probablities)
    return tp[2:x + 2, 2:y + 2].tolist()


def solve_task(task):
    #     bkcolor=-1
    #     for i in len(task[1]):
    #         if i==0:
    #              bkcolor=
    r = getBackground(getCoMat(np.array(task[0][-1])))
    pred1 = solvePatch(np.array(task[0][0]), np.array(task[1][0]), np.array(task[0][-1]), r[1])

    # pred3=[[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    #     if n!=-1:
    #         pred3=solvePatch(np.array(task[0][0]),np.array(task[1][0]),np.array(task[0][-1]),int(n))

    #     elif(r[0]!=0 and r[1]!=0 and r[2]!=0 and np.count_nonzero(np.array(task[0][-1])==0)!=0):
    #         pred3=solvePatch(np.array(task[0][0]),np.array(task[1][0]),np.array(task[0][-1]),0)
    #     else:
    #         pred3=solvePatch(np.array(task[0][0]),np.array(task[1][0]),np.array(task[0][-1]),r[2])

    return pred1


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', -1)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


def solvePatch(t0, t1, t, background):
    to_unique = np.unique(t0)
    t1_unique = np.unique(t1)
    diff = np.setdiff1d(to_unique, t1_unique)
    if t0.shape == t1.shape and len(diff) == 1 and len(to_unique) > len(t1_unique):
        # background=diff[0]
        t = np.array(t)
        background = background
        m, n = t.shape
        tp = np.pad(t, 1, pad_with)
        b = view_as_windows(tp, (3, 3), step=1).reshape(m * n, 3, 3)
        indices = np.arange(m * n).reshape(m, n)
        b_count_old = np.count_nonzero(t == background)
        while np.count_nonzero(t == background) != 0:
            for idx, elem in enumerate(b):
                if np.count_nonzero(elem == background) == 1:
                    i, j = np.where(elem == background)
                    i = i[0]
                    j = j[0]
                    votes = np.zeros(10)
                    for e in b:
                        comparison_list = []
                        for k in range(4):
                            comparison_list.append(np.rot90(e, k))
                        comparison_list.append(np.fliplr(e))
                        comparison_list.append(np.flipud(e))
                        for v in comparison_list:
                            if np.count_nonzero(v == background) == 0 and np.count_nonzero(elem != v) == 1:
                                votes[v[i, j]] += 1
                                break
                    tm, tn = np.where(indices == idx)
                    tm = tm[0]
                    tn = tn[0]
                    if (np.sum(votes) != 0):
                        t[tm + (i - 1), tn + (j - 1)] = np.argmax(votes)
            tp = np.pad(t, 1, pad_with)
            b = view_as_windows(tp, (3, 3), step=1).reshape(m * n, 3, 3)
            b_count_new = np.count_nonzero(t == background)
            if b_count_new == b_count_old:
                break
            b_count_old = b_count_new

        return t.tolist()
    elif t0.shape != t1.shape:
        background = background
        locs = np.where(t == background)
        try:
            imin = locs[0].min()
            imax = locs[0].max()
            jmin = locs[1].min()
            jmax = locs[1].max()
        except:
            return -1
        t = np.array(t)

        # background=getBackground(getCoMat(t))  
        m, n = t.shape
        tp = np.pad(t, 1, pad_with)
        b = view_as_windows(tp, (3, 3), step=1).reshape(m * n, 3, 3)
        indices = np.arange(m * n).reshape(m, n)
        b_count_old = np.count_nonzero(t == background)
        while np.count_nonzero(t == background) != 0:
            for idx, elem in enumerate(b):
                if np.count_nonzero(elem == background) == 1:
                    i, j = np.where(elem == background)
                    i = i[0]
                    j = j[0]
                    votes = np.zeros(10)
                    for e in b:
                        comparison_list = []
                        for k in range(4):
                            comparison_list.append(np.rot90(e, k))
                        comparison_list.append(np.fliplr(e))
                        comparison_list.append(np.flipud(e))
                        for v in comparison_list:
                            if np.count_nonzero(v == background) == 0 and np.count_nonzero(elem != v) == 1:
                                votes[v[i, j]] += 1
                                break
                    tm, tn = np.where(indices == idx)
                    tm = tm[0]
                    tn = tn[0]
                    if (np.sum(votes) != 0):
                        t[tm + (i - 1), tn + (j - 1)] = np.argmax(votes)
            tp = np.pad(t, 1, pad_with)
            b = view_as_windows(tp, (3, 3), step=1).reshape(m * n, 3, 3)
            b_count_new = np.count_nonzero(t == background)
            if b_count_new == b_count_old:
                break
            b_count_old = b_count_new

        return t[imin:imax + 1, jmin:jmax + 1].tolist()

    else:
        return -1


#####
##### END   src_james/ensemble/solvers/Solve_patch.py
#####

#####
##### START src_james/ensemble/solvers/Solve_period.py
#####

import numpy as np

# from src_james.ensemble.colors import mergedict, applyColorMap, checkColorMap, findColorMap
# from src_james.ensemble.period import get_period, same_ratio
# from src_james.ensemble.util import Defensive_Copy


def Solve_period(basic_task):
    # returns -1 if no match is found
    # returns Transformed_Test_Case  if the mathching rule is found
    # for this notebook we only look at mosaics
    Input  = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    Test_Case = Input[-1]
    Input     = Input[:-1]
    colormaps = {}
    if same_ratio(basic_task) == -1:
        return -1
    else:
        R_y, R_X = same_ratio(basic_task)

    for x, y in zip(Input[:-1], Output):

        if get_period(x) == -1:
            return -1
        period_image = get_period(x)
        y_shape = np.zeros((int(len(x) * R_y), int(len(x[0]) * R_X)))
        if len(y_shape) < len(period_image) or len(y_shape[0]) < len(period_image[0]):
            return -1
        y_pred = np.pad(period_image,
                        ((0, len(y_shape) - len(period_image)), (0, len(y_shape[0]) - len(period_image[0]))), "wrap")

        if checkColorMap(y_pred, y):
            colormap = findColorMap(y_pred, y)

            if mergedict([colormaps, colormap]) == False:
                return -1
            colormaps = mergedict([colormaps, colormap])
        else:
            return -1
    if colormaps:
        period_image = get_period(Test_Case)
        y_shape = np.zeros((int(len(Test_Case) * R_y), int(len(Test_Case[0]) * R_X)))
        y_pred  = np.pad(period_image,
                        ((0, len(y_shape) - len(period_image)), (0, len(y_shape[0]) - len(period_image[0]))), "wrap")
        y_pred_final = applyColorMap(y_pred, colormaps)
        return y_pred_final
    else:
        return -1
           

#####
##### END   src_james/ensemble/solvers/Solve_period.py
#####

#####
##### START src_james/ensemble/solvers/Solve_resize.py
#####

import numpy as np

# from src_james.ensemble.colors import checkColorMap, findColorMap, mergedict, applyColorMap
# from src_james.ensemble.resize import resize_o, resize_c
# from src_james.ensemble.split_object import get_bound_image
# from src_james.ensemble.util import Defensive_Copy


def Solve_resize(basic_task):
    Input  = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    Test_Case = Input[-1]
    Input = Input[:-1]
    same_ratio = True
    colormaps  = {}
    R_x = []
    R_y = []
    r1, r2 = 0, 0
    for x, y in zip(Input, Output):
        if len(np.unique(x)) == 1 and np.unique(x)[0] == 0:
            return -1
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
    for x, y in zip(Input, Output):
        x_array = np.array(x)
        y_array = np.array(y)
        if r1 == 0 or r2 == 0:
            return -1
        pre_y = resize_o(x_array, r1, r2)
        if checkColorMap(pre_y, y_array) == False:
            return -1
        else:
            colormap = findColorMap(pre_y, y_array)
        if mergedict([colormaps, colormap]) == False:
            return -1
        colormaps = mergedict([colormaps, colormap])
        pre_y_list = applyColorMap(pre_y, colormaps)
        if pre_y_list != y:
            return -1

    Test_Case_array = np.array(Test_Case)
    return applyColorMap(resize_o(Test_Case_array, r1, r2), colormaps)


def Solve_resize_bound(basic_task):
    Input = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    Test_Case = Input[-1]
    Input = Input[:-1]
    same_ratio = True
    R_x = []
    R_y = []
    colormaps = {}
    r1, r2 = 0, 0
    for x0, y in zip(Input, Output):
        if len(np.unique(x0)) == 1 and np.unique(x0)[0] == 0:
            return -1

        x = get_bound_image(x0)
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
    for x0, y in zip(Input, Output):
        x = get_bound_image(x0)
        x_array = np.array(x)
        y_array = np.array(y)
        if r1 == 0 or r2 == 0:
            return -1
        pre_y = resize_o(x_array, r1, r2)
        if checkColorMap(pre_y, y_array) == False:
            return -1
        else:
            colormap = findColorMap(pre_y, y_array)
        if mergedict([colormaps, colormap]) == False:
            return -1
        colormaps = mergedict([colormaps, colormap])
        pre_y_list = applyColorMap(pre_y, colormaps)
        if pre_y_list != y:
            return -1

    Test_Case_array = np.array(get_bound_image(Test_Case))
    return applyColorMap(resize_o(Test_Case_array, r1, r2), colormaps)


def Solve_resizec(basic_task):
    Input = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    Test_Case = Input[-1]
    Input = Input[:-1]
    colormaps = {}
    for x, y in zip(Input, Output):
        if len(np.unique(x)) == 1 and np.unique(x)[0] == 0:
            return -1
        x_array = np.array(x)
        y_array = np.array(y)

        pre_y = resize_c(x_array)
        if pre_y.shape[0] > 30 or pre_y.shape[1] > 30:
            return -1

        if checkColorMap(pre_y, y_array) == False:
            return -1
        else:
            colormap = findColorMap(pre_y, y_array)
        if mergedict([colormaps, colormap]) == False:
            return -1
        colormaps = mergedict([colormaps, colormap])
        pre_y_list = applyColorMap(pre_y, colormaps)

        if np.shape(pre_y) != np.shape(y_array) or pre_y_list != y:
            return -1

    Test_Case_array = np.array(Test_Case)
    return applyColorMap(resize_c(Test_Case_array), colormaps)


def Solve_resizec_bound(basic_task):
    Input = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    Test_Case = Input[-1]
    Input = Input[:-1]
    colormaps = {}
    for x0, y in zip(Input, Output):
        if len(np.unique(x0)) == 1 and np.unique(x0)[0] == 0:
            return -1
        x = get_bound_image(x0)
        x_array = np.array(x)
        y_array = np.array(y)

        pre_y = resize_c(x_array)
        if pre_y.shape[0] > 30 or pre_y.shape[1] > 30:
            return -1

        if checkColorMap(pre_y, y_array) == False:
            return -1
        else:
            colormap = findColorMap(pre_y, y_array)
        if mergedict([colormaps, colormap]) == False:
            return -1
        colormaps = mergedict([colormaps, colormap])
        pre_y_list = applyColorMap(pre_y, colormaps)

        if np.shape(pre_y) != np.shape(y_array) or pre_y_list != y:
            return -1

    Test_Case_array = np.array(get_bound_image(Test_Case))
    return applyColorMap(resize_c(Test_Case_array), colormaps)


#####
##### END   src_james/ensemble/solvers/Solve_resize.py
#####

#####
##### START src_james/ensemble/solvers/Solve_train_test_map.py
#####

# from src_james.ensemble.colors import checkColorMap, findColorMap, applyColorMap
# from src_james.ensemble.util import Defensive_Copy


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

#####
##### END   src_james/ensemble/solvers/Solve_train_test_map.py
#####

#####
##### START src_james/ensemble/solvers/Solve_trans.py
#####

# def Match_trans(basic_task):
#     #returns -1 if no match is found
#     #returns  Transformed_Test_Case  if the mathching rule is found
#     Input = [Defensive_Copy(x) for x in basic_task[0]]
#     Output = [Defensive_Copy(y) for y in basic_task[1]]
#     Test_Case = Input[-1]
#     Input = Input[:-1]
#     for i in range(len(Geometric)):
#         S = Geometric[i]
#         solved = True
#         for x, y in zip(Input,Output):
#             transformed_x = Apply_geometric(S,x)
#             if transformed_x != y:
#                 solved = False
#                 break
#         if solved == True:
#             Transformed_Test_Case = Apply_geometric(S, Test_Case)
#             return Transformed_Test_Case
#     return -1
# from src_james.ensemble.colors import checkColorMap, findColorMap, mergedict, applyColorMap
# from src_james.ensemble.split_object import get_bound_image
# from src_james.ensemble.transformations import Geometric, Apply_geometric, Glue, Cut
# from src_james.ensemble.util import Defensive_Copy


def Match_trans(basic_task):
    # returns -1 if no match is found
    # returns  Transformed_Test_Case  if the mathching rule is found
    Input  = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    Test_Case = Input[-1]
    Input = Input[:-1]
    colormaps = {}
    colorchage = True
    for i in range(len(Geometric)):
        S = Geometric[i]
        solved = True
        for x, y in zip(Input, Output):
            transformed_x = Apply_geometric(S, x)
            #             if transformed_x != y:
            #                 solved = False
            #                 break
            if checkColorMap(transformed_x, y) == False:
                solved = False
                break
            else:
                colormap = findColorMap(transformed_x, y)
                #                 print(transformed_x,y)
                # print(colormaps,colorchage)
                if colorchage == True:
                    colormaps = mergedict([colormap, colormaps])
                if colormaps == False:
                    colorchage = False
        if solved == True:
            if colorchage:
                Transformed_Test_Case = applyColorMap(Apply_geometric(S, Test_Case), colormaps)
            else:
                Transformed_Test_Case = Apply_geometric(S, Test_Case)
            return Transformed_Test_Case
    return -1


def Solve_trans(basic_task):
    # returns -1 if no match is found
    # returns Transformed_Test_Case  if the mathching rule is found
    # for this notebook we only look at mosaics
    Input = [Defensive_Copy(x) for x in basic_task[0]]
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

            proposed_solution = Match_trans([Input, List])

            if proposed_solution == -1:
                return -1
            else:
                Partial_Solutions.append(proposed_solution)
        Transformed_Test_Case = Glue(Partial_Solutions, r1, r2)
        return Transformed_Test_Case

    return -1


def Match_trans_bound(basic_task):
    # returns -1 if no match is found
    # returns  Transformed_Test_Case  if the mathching rule is found
    Input = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    Test_Case = Input[-1]
    Input = Input[:-1]
    for i in range(len(Geometric)):
        S = Geometric[i]
        solved = True
        for x0, y in zip(Input, Output):
            x = get_bound_image(x0)
            transformed_x = Apply_geometric(S, x)
            if transformed_x != y:
                solved = False
                break
        if solved == True:
            Test_Case_bound = get_bound_image(Test_Case)
            Transformed_Test_Case = Apply_geometric(S, Test_Case_bound)
            return Transformed_Test_Case
    return -1


def Solve_trans_bound(basic_task):
    # returns -1 if no match is found
    # returns Transformed_Test_Case  if the mathching rule is found
    # for this notebook we only look at mosaics
    Input = [Defensive_Copy(x) for x in basic_task[0]]
    Output = [Defensive_Copy(y) for y in basic_task[1]]
    same_ratio = True
    R_x = []
    R_y = []
    for x0, y in zip(Input[:-1], Output):
        x = get_bound_image(x0)
        if x == []:
            same_ratio = False
            break

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

            proposed_solution = Match_trans_bound([Input, List])

            if proposed_solution == -1:
                return -1
            else:
                Partial_Solutions.append(proposed_solution)
        Transformed_Test_Case = Glue(Partial_Solutions, r1, r2)
        return Transformed_Test_Case

    return -1


#####
##### END   src_james/ensemble/solvers/Solve_trans.py
#####

#####
##### START src_james/ensemble/solvers/Solve_trans_negative.py
#####

# from src_james.ensemble.color_classes import take_negative
# from src_james.ensemble.colors import checkColorMap, findColorMap, mergedict, applyColorMap
# from src_james.ensemble.transformations import Cut, Glue, Geometric, Apply_geometric
# from src_james.ensemble.util import Defensive_Copy


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


#####
##### END   src_james/ensemble/solvers/Solve_trans_negative.py
#####

#####
##### START src_james/util/np_cache.py
#####

# Inspired by: https://stackoverflow.com/questions/52331944/cache-decorator-for-numpy-arrays/52332109
from functools import wraps

import numpy as np
from fastcache._lrucache import clru_cache


### Profiler: 2x speedup
def np_cache(maxsize=None, typed=True):
    """
        Decorator:
        @np_cache
        def fn(): return value

        @np_cache(maxsize=128, typed=True)
        def fn(): return value
    """
    maxsize_default=None

    def np_cache_generator(function):

        @wraps(function)
        def wrapper(*args, **kwargs):
            ### def encode(*args, **kwargs):
            args = list(args)  # BUGFIX: TypeError: 'tuple' object does not support item assignment
            for i, arg in enumerate(args):
                if isinstance(arg, np.ndarray):
                    hash = arg.tobytes()
                    if hash not in wrapper.cache:
                        wrapper.cache[hash] = arg
                    args[i] = hash
            for key, arg in kwargs.items():
                if isinstance(arg, np.ndarray):
                    hash = arg.tobytes()
                    if hash not in wrapper.cache:
                        wrapper.cache[hash] = arg
                    kwargs[key] = hash

            return cached_wrapper(*args, **kwargs)

        @clru_cache(maxsize=maxsize, typed=typed)
        def cached_wrapper(*args, **kwargs):
            ### def decode(*args, **kwargs):
            args = list(args)  # BUGFIX: TypeError: 'tuple' object does not support item assignment
            for i, arg in enumerate(args):
                if isinstance(arg, bytes) and arg in wrapper.cache:
                    args[i] = wrapper.cache[arg]
            for key, arg in kwargs.items():
                if isinstance(arg, bytes) and arg in wrapper.cache:
                    kwargs[key] = wrapper.cache[arg]

            return function(*args, **kwargs)

        # copy lru_cache attributes over too
        wrapper.cache       = {}
        wrapper.cache_info  = cached_wrapper.cache_info
        wrapper.cache_clear = cached_wrapper.cache_clear

        return wrapper


    ### def np_cache(maxsize=1024, typed=True):
    if callable(maxsize):
        (function, maxsize) = (maxsize, maxsize_default)
        return np_cache_generator(function)
    else:
        return np_cache_generator

#####
##### END   src_james/util/np_cache.py
#####

#####
##### START src_james/core/CSV.py
#####

import os
import re

import numpy as np

# from src_james.settings import settings


class CSV:
    @classmethod
    def write_submission(cls, dataset: 'Dataset', filename='submission.csv'):
        csv        = CSV.to_csv(dataset)
        line_count = len(csv.split('\n'))
        filename   = os.path.join(settings['dir']['output'], filename)
        with open(filename, 'w') as file:
            file.write(csv)
            print(f"\nwrote: {filename} | {line_count} lines")

    @classmethod
    def object_id(cls, filename, index=0) -> str:
        return re.sub('^.*/|\.json$', '', filename) + '_' + str(index)

    @classmethod
    def to_csv(cls, dataset: 'Dataset'):
        csv = ['output_id,output']
        for task in dataset:
            line = CSV.to_csv_line(task)
            if line: csv.append(line)
        return "\n".join(csv)

    @classmethod
    def to_csv_line(cls, task: 'Task') -> str:
        csv = []
        solutions = set({
            cls.grid_to_csv_string(problem['output'])
            for problem in task['solutions']
        })
        for index, solution_csv in enumerate(solutions):
            if not solution_csv: continue
            line = ",".join([
                cls.object_id(task.filename, index),
                solution_csv
            ])
            csv.append(line)
        return "\n".join(csv)

    # Source: https://www.kaggle.com/c/abstraction-and-reasoning-challenge/overview/evaluation
    # noinspection PyTypeChecker
    @staticmethod
    def grid_to_csv_string(grid: np.ndarray) -> str:
        if grid is None: return None
        grid = np.array(grid).astype('int8').tolist()
        str_pred = str([ row for row in grid ])
        str_pred = str_pred.replace(', ', '')
        str_pred = str_pred.replace('[[', '|')
        str_pred = str_pred.replace('][', '|')
        str_pred = str_pred.replace(']]', '|')
        return str_pred


#####
##### END   src_james/core/CSV.py
#####

#####
##### START src_james/ensemble/features.py
#####

import numpy as np
from fastcache._lrucache import clru_cache

# from src_james.util.np_cache import np_cache

def get_moore_neighbours(color, cur_row, cur_col, nrows, ncols):
    if cur_row <= 0:
        top = -1
    else:
        top = color[cur_row - 1][cur_col]

    if cur_row >= nrows - 1:
        bottom = -1
    else:
        bottom = color[cur_row + 1][cur_col]

    if cur_col <= 0:
        left = -1
    else:
        left = color[cur_row][cur_col - 1]

    if cur_col >= ncols - 1:
        right = -1
    else:
        right = color[cur_row][cur_col + 1]

    return top, bottom, left, right


def get_tl_tr(color, cur_row, cur_col, nrows, ncols):
    if cur_row == 0:
        top_left = -1
        top_right = -1
    else:
        if cur_col == 0:
            top_left = -1
        else:
            top_left = color[cur_row - 1][cur_col - 1]
        if cur_col == ncols - 1:
            top_right = -1
        else:
            top_right = color[cur_row - 1][cur_col + 1]

    return top_left, top_right


def make_features(input_color, nfeat=13, local_neighb = 5):
    nrows, ncols = input_color.shape
    feat = np.zeros((nrows * ncols, nfeat))
    cur_idx = 0
    for i in range(nrows):
        for j in range(ncols):
            feat[cur_idx, 0]   = i
            feat[cur_idx, 1]   = j
            feat[cur_idx, 2]   = input_color[i][j]
            feat[cur_idx, 3:7] = get_moore_neighbours(input_color, i, j, nrows, ncols)
            feat[cur_idx, 7:9] = get_tl_tr(input_color, i, j, nrows, ncols)
            feat[cur_idx, 9]   = len(np.unique(input_color[i, :]))
            feat[cur_idx, 10]  = len(np.unique(input_color[:, j]))
            feat[cur_idx, 11]  = (i + j)
            feat[cur_idx, 12]  = len(np.unique(
                input_color[i - local_neighb:i + local_neighb,
                            j - local_neighb:j + local_neighb])
            )

            cur_idx += 1
    return feat


def features(task, mode='train', nfeat=13):
    num_train_pairs = len(task[mode])
    feat, target = [], []

    global local_neighb
    for task_num in range(num_train_pairs):
        input_color = np.array(task[mode][task_num]['input'])
        # print(input_color)
        target_color = task[mode][task_num]['output']
        # print(target_color)
        nrows, ncols = len(task[mode][task_num]['input']), len(task[mode][task_num]['input'][0])

        target_rows, target_cols = len(task[mode][task_num]['output']), len(task[mode][task_num]['output'][0])

        if (target_rows != nrows) or (target_cols != ncols):
            print('Number of input rows:', nrows, 'cols:', ncols)
            print('Number of target rows:', target_rows, 'cols:', target_cols)
            not_valid = 1
            return None, None, 1

        imsize = nrows * ncols
        # offset = imsize*task_num*3 #since we are using three types of aug
        feat.extend(make_features(input_color, nfeat))
        target.extend(np.array(target_color).reshape(-1, ))

    return np.array(feat), np.array(target), 0


#####
##### END   src_james/ensemble/features.py
#####

#####
##### START src_james/ensemble/sample_sub/sample_sub1.py
#####

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import json
import os

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

# mode = 'eval'
# from src_james.core.CSV import CSV
# from src_james.ensemble.features import features, make_features
# from src_james.ensemble.sample_sub.path import task_path, mode, data_path, output_dir

all_task_ids = sorted(os.listdir(task_path))

nfeat = 13
local_neighb = 5
valid_scores = {}

model_accuracies = {'ens': []}
pred_taskids = []


sample_sub1 = pd.read_csv(data_path/'sample_submission.csv')
sample_sub1 = sample_sub1.set_index('output_id', drop=False)

for task_id in all_task_ids:
    task_file = str(task_path / task_id)
    with open(task_file, 'r') as f:
        task = json.load(f)

    feat, target, not_valid = features(task)
    if not_valid:
        print('ignoring task', task_file)
        print()
        not_valid = 0
        continue

    xgb =  XGBClassifier(n_estimators=10, n_jobs=-1)
    xgb.fit(feat, target, verbose=-1)


    #     training on input pairs is done.
    #     test predictions begins here

    num_test_pairs = len(task['test'])
    for task_num in range(num_test_pairs):
        cur_idx = 0
        input_color = np.array(task['test'][task_num]['input'])
        nrows, ncols = len(task['test'][task_num]['input']), len(
            task['test'][task_num]['input'][0])
        feat = make_features(input_color, nfeat)

        print('Made predictions for ', task_id[:-5])

        preds = xgb.predict(feat).reshape(nrows,ncols)

        if (mode=='train') or (mode=='eval'):
            ens_acc = (np.array(task['test'][task_num]['output'])==preds).sum()/(nrows*ncols)

            model_accuracies['ens'].append(ens_acc)

            pred_taskids.append(f'{task_id[:-5]}_{task_num}')

        #             print('ensemble accuracy',(np.array(task['test'][task_num]['output'])==preds).sum()/(nrows*ncols))
        #             print()

        preds = preds.astype(int).tolist()
        #         plot_test(preds, task_id)
        sample_sub1.loc[f'{task_id[:-5]}_{task_num}',
                        'output'] = CSV.grid_to_csv_string(preds)



if (mode=='train') or (mode=='eval'):
    df = pd.DataFrame(model_accuracies, index=pred_taskids)
    print(df.head(10))

    print(df.describe())
    for c in df.columns:
        print(f'for {c} no. of complete tasks is', (df.loc[:, c]==1).sum())

    df.to_csv('ens_acc.csv')



sample_sub1.head()
sample_sub1.to_csv(output_dir/'submission1.csv', index=False)

#####
##### END   src_james/ensemble/sample_sub/sample_sub1.py
#####

#####
##### START src_james/ensemble/sample_sub/sample_sub2.py
#####

import json

import pandas as pd

# from src_james.ensemble.sample_sub.example_grid import example_grid
# from src_james.ensemble.sample_sub.path import test_path, data_path, output_dir
# from src_james.ensemble.solvers.Match_crop_mode import Match_crop_mode, Match_crop_mode_1
# from src_james.ensemble.solvers.Recolor import Recolor
# from src_james.ensemble.solvers.Recolor0 import Recolor0, Recolor0_bound
# from src_james.ensemble.solvers.Solve_color_crop import Solve_color_crop
# from src_james.ensemble.solvers.Solve_connect import solve_connect
# from src_james.ensemble.solvers.Solve_filing import Solve_filling
# from src_james.ensemble.solvers.Solve_inoutmap import solve_inoutmap, solve_inoutmap_colormap
# from src_james.ensemble.solvers.Solve_logtic import Solve_logtic
# from src_james.ensemble.solvers.Solve_mode_dict import Solve_mode_dict
# from src_james.ensemble.solvers.Solve_mul_color_bound import Solve_mul_color_bound
# from src_james.ensemble.solvers.Solve_negative import Solve_negative
# from src_james.ensemble.solvers.Solve_object_mode_color import solve_object_mode, solve_object_mode_color
# from src_james.ensemble.solvers.Solve_output_color_change_mode import solve_output_color_change_mode
# from src_james.ensemble.solvers.Solve_patch import solve_task
# from src_james.ensemble.solvers.Solve_period import Solve_period
# from src_james.ensemble.solvers.Solve_resize import Solve_resize, Solve_resizec, Solve_resize_bound, Solve_resizec_bound
# from src_james.ensemble.solvers.Solve_train_test_map import Solve_train_test_map
# from src_james.ensemble.solvers.Solve_trans import Solve_trans_bound, Solve_trans
# from src_james.ensemble.solvers.Solve_trans_negative import Solve_trans_negative
# from src_james.ensemble.util import Create, flattener

sample_sub2 = pd.read_csv(data_path/'sample_submission.csv')
sample_sub2 = sample_sub2.set_index('output_id', drop=False)
sample_sub2.head()

Solved = []
Problems = sample_sub2['output_id'].values
Proposed_Answers = []

for i in range(len(Problems)):
    # print(i)
    preds=[example_grid,example_grid,example_grid]
    predict_solution=[]
    output_id = Problems[i]
    task_id = output_id.split('_')[0]
    pair_id = int(output_id.split('_')[1])
    f = str(test_path / str(task_id + '.json'))

    with open(f, 'r') as read_file:
        task = json.load(read_file)

    basic_task = Create(task, pair_id)

    try:
        predict_solution.append(Solve_mode_dict(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(Recolor0(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(Recolor0_bound(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(Recolor(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(Solve_train_test_map(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(Solve_color_crop(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(Solve_filling(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(Solve_trans_bound(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(Solve_trans(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(Solve_mul_color_bound(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(Solve_period(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(solve_inoutmap(basic_task,0,0,0,0))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(solve_inoutmap_colormap(basic_task,1,1,1,1))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(solve_inoutmap_colormap(basic_task,2,2,2,2))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(Solve_logtic(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(solve_task(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(Solve_negative(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(Solve_trans_negative(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(solve_object_mode(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(solve_object_mode_color(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(solve_output_color_change_mode(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(Solve_resize(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(Solve_resizec(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(Solve_resize_bound(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(Solve_resizec_bound(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(solve_connect(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(Match_crop_mode(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(Match_crop_mode_1(basic_task))
    except:
        predict_solution.append(-1)


    for j in range(len(predict_solution)):
        if predict_solution[j]!=-1 and predict_solution[j] not in preds:
            preds.append(predict_solution[j])


    pred = ''
    if len(preds)>3:
        Solved.append(i)
        pred1 = flattener(preds[-1])
        pred2 = flattener(preds[-2])
        pred3 = flattener(preds[-3])
        pred  = pred+pred1+' '+pred2+' '+pred3+' '

    if pred == '':
        pred = flattener(example_grid)

    Proposed_Answers.append(pred)

sample_sub2['output'] = Proposed_Answers
sample_sub2.to_csv(output_dir/'submission2.csv', index=False)

#####
##### END   src_james/ensemble/sample_sub/sample_sub2.py
#####

#####
##### START src_james/ensemble/sample_sub/sample_sub4.py
#####

import json

import pandas as pd

# from src_james.ensemble.sample_sub.example_grid import example_grid
# from src_james.ensemble.sample_sub.path import data_path, test_path, output_dir
# from src_james.ensemble.solvers.Solve_split_shape import Solve_mul_color_bound_negative, Solve_juxt, Solve_split_shape
# from src_james.ensemble.solvers.Solve_split_shape import Solve_split_shape_negative, solve_cross_map
# from src_james.ensemble.solvers.Solve_split_shape import solve_cross_map_line, Solve_mul_color_negative
# from src_james.ensemble.util import Create, flattener

sample_sub4 = pd.read_csv(data_path / 'sample_submission.csv')
sample_sub4.head()

# display(example_grid)
print(flattener(example_grid))

Solved = []
Problems = sample_sub4['output_id'].values
Proposed_Answers = []

for i in range(len(Problems)):
    preds = [example_grid, example_grid, example_grid]
    predict_solution = []
    output_id = Problems[i]
    task_id = output_id.split('_')[0]
    pair_id = int(output_id.split('_')[1])
    f = str(test_path / str(task_id + '.json'))

    with open(f, 'r') as read_file:
        task = json.load(read_file)

    basic_task = Create(task, pair_id)
    try:
        predict_solution.append(Solve_split_shape_negative(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(solve_cross_map(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(solve_cross_map_line(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(Solve_mul_color_negative(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(Solve_mul_color_bound_negative(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(Solve_juxt(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(Solve_split_shape(basic_task))
    except:
        predict_solution.append(-1)

    for j in range(len(predict_solution)):
        if predict_solution[j] != -1 and predict_solution[j] not in preds:
            preds.append(predict_solution[j])

    pred = ''
    if len(preds) > 3:
        Solved.append(i)
        pred1 = flattener(preds[-1])
        pred2 = flattener(preds[-2])
        pred3 = flattener(preds[-3])
        pred = pred + pred1 + ' ' + pred2 + ' ' + pred3 + ' '

    if pred == '':
        pred = flattener(example_grid)

    Proposed_Answers.append(pred)

sample_sub4['output'] = Proposed_Answers
sample_sub4.to_csv(output_dir/'submission4.csv', index=False)

#####
##### END   src_james/ensemble/sample_sub/sample_sub4.py
#####

#####
##### START src_james/ensemble/sample_sub/sample_sub_combine.py
#####

#####
# from src_james.ensemble.sample_sub.path import output_dir
# from src_james.ensemble.sample_sub.sample_sub1 import sample_sub1
# from src_james.ensemble.sample_sub.sample_sub2 import sample_sub2
# from src_james.ensemble.sample_sub.sample_sub4 import sample_sub4

sample_sub1 = sample_sub1.reset_index(drop=True).sort_values(by="output_id")
sample_sub2 = sample_sub2.set_index('output_id', drop=True).sort_values(by="output_id")
# sample_sub3 = sample_sub3.set_index('output_id', drop=True).sort_values(by="output_id")
sample_sub4 = sample_sub4.reset_index(drop=True).sort_values(by="output_id")

out1 = sample_sub1["output"].astype(str).values
out2 = sample_sub2["output"].astype(str).values
# out3 = sample_sub3["output"].astype(str).values
out4 = sample_sub4["output"].astype(str).values
merge_output = []

# for o1, o4 in zip(out1, out4):
#     o = o1.strip().split(" ")[:1] + o4.strip().split(" ")[:1]
for o1, o2, o4 in zip(out1, out2, out4):
    o = o1.strip().split(" ")[:1] + o2.strip().split(" ")[:1] + o4.strip().split(" ")[:1]
    o = " ".join(o[:3])
    merge_output.append(o)

sample_sub1["output"] = merge_output
sample_sub1["output"] = sample_sub1["output"].astype(str)
sample_sub1.to_csv(output_dir/"submission.csv", index=False)

#####
##### END   src_james/ensemble/sample_sub/sample_sub_combine.py
#####

##### 
##### ./submission/kaggle_compile.py src_james/ensemble/sample_sub/sample_sub_combine.py
##### 
##### 2020-05-24 02:50:55+01:00
##### 
##### origin	git@github.com:seshurajup/kaggle-arc.git (fetch)
##### origin	git@github.com:seshurajup/kaggle-arc.git (push)
##### 
#####   james-wip c81cf89 Solvers | work in progress - broken
##### * master    7792251 Ensemble | import functions from kaggle notebook
##### 
##### 7792251fffe4e87da4c583afa466e3337c92b1d0
##### 
