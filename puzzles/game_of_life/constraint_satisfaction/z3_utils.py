import itertools
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import z3
from fastcache import clru_cache


### Get Neighbourhood Function

@clru_cache(None)
def get_neighbourhood_coords(shape: Tuple[int,int], x: int, y: int, distance=1) -> List[Tuple[int,int]]:
    output = []
    for dx, dy in itertools.product(range(-distance,distance+1), range(-distance,distance+1)):
        if dx == dy == 0: continue      # ignore self
        nx = (x + dx) % shape[0]        # modulo loop = board wraparound
        ny = (y + dy) % shape[1]
        output.append( (nx, ny) )
    return output


def get_neighbourhood_cells(cells: Union[np.ndarray, List[List[int]]], x: int, y: int, distance=1) -> List[int]:
    shape  = ( len(cells), len(cells[0]) )
    coords = get_neighbourhood_coords(shape, x, y, distance)
    output = [ cells[x][y] for (x,y) in coords ]
    return output



### Casting Function

def solver_to_numpy_3d(z3_solver, t_cells) -> np.ndarray:  # np.int8[time][x][y]
    is_sat = z3_solver.check() == z3.sat
    output = np.array([
        [
            [
                int(z3.is_true(z3_solver.model()[cell])) if is_sat else 0
                for y, cell in enumerate(cells)
            ]
            for x, cells in enumerate(t_cells[t])
        ]
        for t in range(len(t_cells))
    ], dtype=np.int8)
    return output
