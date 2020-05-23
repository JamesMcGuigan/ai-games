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
