from src_james.solver_multimodel.Solver import Solver
from src_james.solver_multimodel.functions import *


class BorderSolver(Solver):
    verbose = True
    debug = True
    cache = {}
    queries = [
        *range(0,10),
        max_color,     # FAIL: evaluation/fc754716.json
        max_color_1d,
        min_color,
        min_color_1d,
        count_colors,
        count_squares,
        np.count_nonzero,
    ]

    def task_has_border(self, task):
        if not self.is_task_shape_ratio_consistant(task): return False
        return all([ self.grid_has_border(spec['output']) for spec in task['train'] ])

    def grid_has_border(self, grid):
        if min(grid.shape) <= 2: return False  # single color problem

        grid_center = grid[1:-1,1:-1]
        return np.count_nonzero(grid_center) == 0 and all([
            np.count_nonzero(border) == len(border)
            for border in [ grid[0,:], grid[-1,:], grid[:,0], grid[:,-1] ]
        ])

    def detect(self, task):
        return self.task_has_border(task)

    def test(self, task):
        if task.filename in self.cache: return True
        for query in self.queries:
            args = [ query ]
            if self.is_lambda_valid(task, self.action, *args, task=task):
                self.cache[task.filename] = args
                if self.verbose: print(self.action, args)
                return True
        return False

    def action(self, grid, query=None, task=None):
        ratio  = list(self.task_shape_ratios(task))[0]
        output = np.zeros(( int(grid.shape[0] * ratio[0]), int(grid.shape[1] * ratio[1]) ))
        color  = query(grid) if callable(query) else query
        output[:,:] = color
        output[1:-1,1:-1] = 0
        return output
