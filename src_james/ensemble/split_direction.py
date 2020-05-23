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
