
# BROKEN?

import numpy as np

from src.functions.queries.grid import grid_unique_colors
from src.util.np_cache import np_cache



@np_cache
def grid_invert_color(grid: np.ndarray):
    colors = grid_unique_colors(grid)
    if len(colors) != 2:
        output = np.zeros(grid.shape, dtype=np.int8)
        return output
    else:
        color1 = colors[0]
        color2 = colors[1]
        mask   = grid[ grid == color1 ]
        output = np.full(grid.shape, color1, dtype=np.int8)
        output[mask] = color2
        return output


