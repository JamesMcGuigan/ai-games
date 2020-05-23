
# BROKEN?
from src_james.solver_multimodel.queries.grid import max_color
from src_james.util.np_cache import np_cache


@np_cache
def invert(grid, color=None):
    if callable(color): color = color(grid)
    if color is None:   color = max_color(grid)
    if color:
        grid        = grid.copy()
        mask_zero   = grid[ grid == 0 ]
        mask_square = grid[ grid != 0 ]
        grid[mask_zero]   = color
        grid[mask_square] = 0
    return grid
