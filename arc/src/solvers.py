from typing import List

from src.solver_multimodel.core.Solver import Solver
from src.solver_multimodel.solvers.BorderSolver import BorderSolver
from src.solver_multimodel.solvers.DoNothingSolver import DoNothingSolver
from src.solver_multimodel.solvers.GeometrySolver import GeometrySolver
from src.solver_multimodel.solvers.GlobSolver import GlobSolver
from src.solver_multimodel.solvers.SingleColorSolver import SingleColorSolver
from src.solver_multimodel.solvers.TessellationSolver import TessellationSolver
from src.solver_multimodel.solvers.XGBGridSolver import XGBGridSolver
from src.solver_multimodel.solvers.XGBSingleColorSolver import XGBSingleColorSolver
from src.solver_multimodel.solvers.ZoomSolver import ZoomSolver

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
