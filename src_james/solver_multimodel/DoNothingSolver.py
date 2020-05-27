from src_james.solver_multimodel.Solver import Solver



class DoNothingSolver(Solver):
    def solve_grid(self, grid, task=None, *args):
        return grid
