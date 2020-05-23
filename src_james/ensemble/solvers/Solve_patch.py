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
