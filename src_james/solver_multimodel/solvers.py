from typing import List

from src_james.solver_multimodel.BorderSolver import BorderSolver
from src_james.solver_multimodel.DoNothingSolver import DoNothingSolver
from src_james.solver_multimodel.GeometrySolver import GeometrySolver
from src_james.solver_multimodel.SingleColorSolver import SingleColorSolver
from src_james.solver_multimodel.Solver import Solver
from src_james.solver_multimodel.TessellationSolver import TessellationSolver
from src_james.solver_multimodel.XGBSolver import XGBSolver
from src_james.solver_multimodel.ZoomSolver import ZoomSolver

solvers: List[Solver] = [
    DoNothingSolver(),
    BorderSolver(),
    GeometrySolver(),
    SingleColorSolver(),
    ZoomSolver(),
    TessellationSolver(),
    XGBSolver(),
]
