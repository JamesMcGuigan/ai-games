from src_james.core.DataModel import Task
from src_james.settings import settings
from src_james.solver_multimodel.queries.colors import task_is_singlecolor
from src_james.solver_multimodel.queries.grid import *
from src_james.solver_multimodel.queries.ratio import task_shape_ratios
from src_james.solver_multimodel.Solver import Solver


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

    def detect(self, task):
        return task_is_singlecolor(task)

    def fit(self, task: Task):
        if task.filename in self.cache: return True
        for query in self.queries:
            args = ( query, )
            if self.is_lambda_valid(task, self.action, *args, task=task):
                self.cache[task.filename] = args
                break

    def action(self, grid, query=None, task=None):
        ratio  = task_shape_ratios(task)[0]
        output = np.zeros(( int(grid.shape[0] * ratio[0]), int(grid.shape[1] * ratio[1]) ), dtype=np.int8)
        color  = query(grid) if callable(query) else query
        output[:,:] = color
        return output




if __name__ == '__main__' and not settings['production']:
    solver = SingleColorSolver()
    solver.verbose = True
    filenames = [
        'training/5582e5ca.json',  # solved
        'training/445eab21.json',  # solved
        'training/27a28665.json',
        'training/44f52bb0.json',
        'evaluation/3194b014.json',
        'test/3194b014.json',
    ]
    for filename in filenames:
        task = Task(filename)
        solver.plot_detects([task])

    # competition = Competition()
    # # competition['test'].apply(solver.solve_dataset)
    # competition.map(solver.solve_dataset)
    # print(competition)