from typing import List

from src_james.solver_multimodel.BorderSolver import BorderSolver
from src_james.solver_multimodel.DoNothingSolver import DoNothingSolver
from src_james.solver_multimodel.GeometrySolver import GeometrySolver
from src_james.solver_multimodel.GlobSolver import GlobSolver
from src_james.solver_multimodel.SingleColorSolver import SingleColorSolver
from src_james.solver_multimodel.Solver import Solver
from src_james.solver_multimodel.TessellationSolver import TessellationSolver
from src_james.solver_multimodel.XGBGridSolver import XGBGridSolver
from src_james.solver_multimodel.XGBSingleColorSolver import XGBSingleColorSolver
from src_james.solver_multimodel.ZoomSolver import ZoomSolver

solvers: List[Solver] = [
    # Deterministic (all solved answers are correct)
    GlobSolver(),
    DoNothingSolver(),
    BorderSolver(),
    GeometrySolver(),
    SingleColorSolver(),
    ZoomSolver(),
    TessellationSolver(),

    # Non-Deterministic (lots of random guesses)
    XGBSingleColorSolver(),
    XGBGridSolver(),
    # XGBGridSolverDart(),     # These don't provide any additional value
    # XGBGridSolverGBtree(),
    # XGBGridSolverGBlinear(),
]
