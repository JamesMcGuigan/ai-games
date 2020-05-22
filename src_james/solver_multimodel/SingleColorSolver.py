from src_james.solver_multimodel.Solver import Solver
from src_james.solver_multimodel.functions import *


class SingleColorSolver(Solver):
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

    def grid_unique_colors(self, grid):
        return np.unique(grid.flatten())

    def task_is_singlecolor(self, task):
        if not self.is_task_shape_ratio_consistant(task): return False
        return all([ len(self.grid_unique_colors(spec['output'])) == 1 for spec in task['train'] ])

    def detect(self, task):
        return self.task_is_singlecolor(task)

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
        return output
