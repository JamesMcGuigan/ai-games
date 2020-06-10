import numpy as np


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


def make_features(input_color, nfeat=13, local_neighb=5):
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


def features(task, mode='train', nfeat=13, local_neighb=5):
    num_train_pairs = len(task[mode])
    feat, target = [], []

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
