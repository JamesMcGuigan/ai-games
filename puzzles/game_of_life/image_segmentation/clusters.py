# Source: https://www.kaggle.com/jamesmcguigan/game-of-life-image-segmentation-solver
from typing import List

import numpy as np
import scipy
import scipy.ndimage
import scipy.sparse
import skimage
import skimage.measure

from image_segmentation.tessellation import detessellate_board
from image_segmentation.tessellation import tessellate_board


def label_board(board):
    """  """
    tessellation = tessellate_board(board)
    tessellation = scipy.ndimage.convolve(tessellation, [[0,1,0],[1,1,1],[0,1,0]]).astype(np.bool).astype(np.int8)
    labeled = skimage.measure.label(tessellation, background=0, connectivity=2)
    labeled = detessellate_board(labeled)
    return labeled


def extract_clusters(board: np.ndarray) -> List[np.ndarray]:
    labeled  = label_board(board)
    return extract_clusters_from_labels(board, labeled)


def extract_clusters_from_labels(board: np.ndarray, labeled: np.ndarray) -> List[np.ndarray]:
    labels   = np.unique(labeled)
    clusters = []
    for label in labels:
        # if label == 0: continue  # preserve index order with labels
        cluster = board * ( labeled == label )
        clusters.append(cluster)
    return clusters
