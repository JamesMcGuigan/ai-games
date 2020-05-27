from typing import List

from src_james.solver_multimodel.BorderSolver import BorderSolver
from src_james.solver_multimodel.DoNothingSolver import DoNothingSolver
from src_james.solver_multimodel.GeometrySolver import GeometrySolver
from src_james.solver_multimodel.GlobSolver import GlobSolver
from src_james.solver_multimodel.SingleColorSolver import SingleColorSolver
from src_james.solver_multimodel.Solver import Solver
from src_james.solver_multimodel.TessellationSolver import TessellationSolver
from src_james.solver_multimodel.XGBGridSolver import XGBGridSolverDart
from src_james.solver_multimodel.XGBGridSolver import XGBGridSolverGBlinear
from src_james.solver_multimodel.XGBGridSolver import XGBGridSolverGBtree
from src_james.solver_multimodel.ZoomSolver import ZoomSolver

solvers: List[Solver] = [
    GlobSolver(),
    DoNothingSolver(),
    BorderSolver(),
    GeometrySolver(),
    SingleColorSolver(),
    ZoomSolver(),
    TessellationSolver(),
    XGBGridSolverDart(),
    XGBGridSolverGBtree(),
    XGBGridSolverGBlinear(),
]
