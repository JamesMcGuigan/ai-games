from src.solver_multimodel.core.Solver import Solver


class DoNothingSolver(Solver):
    def solve_grid(self, grid, task=None, *args):
        return grid
