from src_james.core.DataModel import Competition
from src_james.core.DataModel import Problem
from src_james.core.DataModel import ProblemSet
from src_james.core.DataModel import Task
from src_james.plot import plot_task
from src_james.settings import settings
from src_james.solver_multimodel.queries.colors import task_is_singlecolor
from src_james.solver_multimodel.queries.grid import *
from src_james.solver_multimodel.queries.ratio import is_task_output_grid_shape_constant
from src_james.solver_multimodel.queries.ratio import task_output_grid_shape
from src_james.solver_multimodel.Solver import Solver
from src_james.solver_multimodel.transforms.singlecolor import identity
from src_james.solver_multimodel.transforms.singlecolor import unique_colors_sorted
from src_james.solver_multimodel.XGBEncoder import XGBEncoder


class SingleColorXGBEncoder(XGBEncoder):
    dtype    = np.int8,
    encoders = {
        np.array: [ identity ],
    }
    features = {
        np.array:   [
            unique_colors_sorted,
            # max_color,
            # max_color_1d,
            # min_color,
            # min_color_1d,
            # np_bincount,
            # np_hash,
            # np_shape,
            # count_colors,
            # count_squares,
        ],
        ProblemSet: [],
        Problem:    [],
        Task:       [
            # task_output_unique_sorted_colors
        ]
    }
    def __init__(self, encoders=None, features=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoders = encoders if encoders is not None else self.__class__.encoders
        self.features = features if features is not None else self.__class__.features


# BUG: XGBoost only works if the output colors have already been seen in the input
class XGBSingleColorSolver(Solver):
    verbose = True
    debug   = True
    cache   = {}

    def detect(self, task):
        return all([
            task_is_singlecolor(task),
            is_task_output_grid_shape_constant(task)  # TODO: OutputGridSizeSolver
        ])


    def fit(self, task: Task):
        if task.filename in self.cache: return True

        encoder = SingleColorXGBEncoder(output_encoder=max_color)
        encoder.fit(task)
        color   = encoder.predict(task)
        self.cache[task.filename] = (color,) if color is not None else None


    def action(self, encoder=None, task=None, **kwargs):
        if task.filename not in self.cache:   self.fit(task)
        if self.cache[task.filename] is None: return None

        color       = self.cache[task.filename][0]
        output_size = task_output_grid_shape()
        if not output_size: return None
        if color is None:   return None

        output = np.full(output_size, fill_value=color)
        return output



if __name__ == '__main__' and not settings['production']:
    solver = XGBSingleColorSolver()
    solver.verbose = True
    filenames = [
        'training/5582e5ca.json',  # solved
        # 'training/445eab21.json',  # solved
        # 'training/27a28665.json',
        # 'training/44f52bb0.json',
        # 'evaluation/3194b014.json',
        # 'test/3194b014.json',
    ]
    for filename in filenames:
        task = Task(filename)
        plot_task(task)
        solver.plot(task)

    competition = Competition()
    # competition['test'].apply(solver.solve_dataset)
    competition.map(solver.solve_dataset)
    print(competition)