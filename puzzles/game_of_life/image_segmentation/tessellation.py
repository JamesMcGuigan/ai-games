# Source: https://www.kaggle.com/jamesmcguigan/game-of-life-image-segmentation-solver
import itertools
from itertools import product

import numpy as np


def tessellate_board(board):
    """ Create a 75x75 (3x) tesselation of the board to account for edge objects """
    shape        = board.shape
    tessellation = np.zeros((shape[0]*3, shape[1]*3), dtype=np.int8)
    for x,y in product( range(3), range(3) ):
        tessellation[ shape[0]*x : shape[0]*(x+1), shape[1]*y : shape[1]*(y+1) ] = board
    return tessellation


def detessellate_board(tessellation):
    """ Merge 3x tesselation back into 25x25 grid, by working out sets of overlapping regions """
    shape = tessellation.shape[0] // 3, tessellation.shape[1] // 3
    views = np.stack([
        tessellation[ shape[0]*x : shape[0]*(x+1), shape[1]*y : shape[1]*(y+1) ].flatten()
        for x,y in product( range(3), range(3) )
    ])
    cells = [ set(views[:,n]) - {0} for n in range(len(views[0])) ]
    for cell1, cell2 in itertools.product(cells, cells):
        if cell1 & cell2:
            cell1 |= cell2  # merge overlapping regions
            cell2 |= cell1
    cells  = np.array([ min(cell) if cell else 0 for cell in cells ])
    labels = sorted(set(cells))
    cells  = np.array([ labels.index(cell) for cell in cells ])  # map back to sequential numbers
    return cells.reshape(shape)
