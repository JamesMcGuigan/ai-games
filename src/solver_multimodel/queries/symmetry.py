import numpy as np

from src.util.np_cache import np_cache


@np_cache()
def is_grid_symmetry(grid) -> bool:
    return (
        is_grid_symmetry_horz(grid)
     or is_grid_symmetry_vert(grid)
     or is_grid_symmetry_rot90(grid)
     or is_grid_symmetry_rot180(grid)
     or is_grid_symmetry_transpose(grid)
    )

@np_cache()
def is_grid_symmetry_horz(grid) -> bool:
    return np.array_equal(grid, np.flip(grid, 0))

@np_cache()
def is_grid_symmetry_vert(grid) -> bool:
    return np.array_equal(grid, np.flip(grid, 1))

@np_cache()
def is_grid_symmetry_rot90(grid) -> bool:
    return np.array_equal(grid, np.rot90(grid))

@np_cache()
def is_grid_symmetry_rot180(grid) -> bool:
    return np.array_equal(grid, np.rot90(grid,2))

def is_grid_symmetry_transpose(grid) -> bool:
    return np.array_equal(grid, np.transpose(grid))