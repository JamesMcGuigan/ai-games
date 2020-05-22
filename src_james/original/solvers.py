from typing import List

from src_james.original.BorderSolver import BorderSolver
from src_james.original.DoNothingSolver import DoNothingSolver
from src_james.original.GeometrySolver import GeometrySolver
from src_james.original.SingleColorSolver import SingleColorSolver
from src_james.original.Solver import Solver
from src_james.original.TessellationSolver import TessellationSolver
from src_james.original.ZoomSolver import ZoomSolver

solvers: List[Solver] = [
    DoNothingSolver(),
    BorderSolver(),
    GeometrySolver(),
    SingleColorSolver(),
    TessellationSolver(),
    ZoomSolver(),
]
