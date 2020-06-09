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
