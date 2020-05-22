from typing import List

from .BorderSolver import BorderSolver
from .DoNothingSolver import DoNothingSolver
from .GeometrySolver import GeometrySolver
from .SingleColorSolver import SingleColorSolver
from .Solver import Solver
from .TessellationSolver import TessellationSolver



# from .ZoomSolver import ZoomSolver



solvers: List[Solver] = [
    DoNothingSolver(),
    BorderSolver(),
    GeometrySolver(),
    SingleColorSolver(),
    TessellationSolver(),
    # ZoomSolver(),
]
