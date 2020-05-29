from typing import List
from typing import Union

from src.core.DataModel import Competition
from src.core.DataModel import Problem
from src.core.DataModel import ProblemSet
from src.core.DataModel import Task
from src.plot import plot_task
from src.settings import settings
from src.solver_multimodel.queries.colors import task_is_singlecolor
from src.solver_multimodel.queries.grid import *
from src.solver_multimodel.queries.ratio import is_task_output_grid_shape_constant
from src.solver_multimodel.queries.ratio import task_output_grid_shape
from src.solver_multimodel.transforms.singlecolor import identity
from src.solver_multimodel.transforms.singlecolor import np_bincount
from src.solver_multimodel.transforms.singlecolor import np_hash
from src.solver_multimodel.transforms.singlecolor import np_shape
from src.solver_multimodel.transforms.singlecolor import unique_colors_sorted
from src.solver_multimodel.XGBEncoder import ProblemSetSolver
from src.solver_multimodel.XGBEncoder import XGBEncoder


class SingleColorXGBEncoder(XGBEncoder):
    dtype    = np.int8,
    encoders = {
        np.array: [ identity ],
    }
    features = {
        np.array:   [
            unique_colors_sorted,
            max_color,
            max_color_1d,
            min_color,
            min_color_1d,
            np_bincount,
            np_hash,
            np_shape,
            count_colors,
            count_squares,
        ],
        ProblemSet: [],
        Problem:    [],
        Task:       [
            # task_output_unique_sorted_colors
        ]
    }
    encoder_defaults = {
        **XGBEncoder.encoder_defaults,
        'max_delta_step':   np.inf,  # unsure if this has an effect
        'max_depth':        1,       # possibly required for this problem
        'n_estimators':     1,       # possibly required for this problem
        'min_child_weight': 0,       # possibly required for this problem
        # 'max_delta_step':   1,
        # 'objective':       'rank:map',
        # 'objective':       'reg:squarederror',
        # 'max_delta_step':   1,
        # 'n_jobs':          -1,
    }
    def __init__(self, encoders=None, features=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoders = encoders if encoders is not None else self.__class__.encoders
        self.features = features if features is not None else self.__class__.features


    ### DEBUG
    # def predict(self,
    #             problemset: Union[ProblemSet, Task],
    #             xgb:   XGBClassifier = None,
    #             task:  Task = None,
    #             *args, **kwargs
    # ) -> Any:
    #     task       = task or (problemset if isinstance(problemset, Task) else problemset.task)
    #     problemset = (problemset['test'] if isinstance(problemset, Task) else problemset )
    #     if task.filename not in self.cache:   self.fit(task)
    #     # if self.cache[task.filename] is None: return None  # Unsolvable mapping
    #
    #     output = [ 8 ] * len(task['test'])
    #     return output



# BUG: XGBoost only works if the output colors have already been seen in the input
class XGBSingleColorSolver(ProblemSetSolver):
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
        colors  = encoder.predict(task)
        self.cache[task.filename] = colors

    ### DEBUG
    # def test(self, task: Task = None) -> Any:
    #     return True

    # BUGFIX: TypeError: solve_grid() got multiple values for argument 'task'
    def predict(self, problemset: Union[ProblemSet,Task], *args, task: Task=None, **kwargs) -> Union[None,List[np.ndarray]]:
        task       = task or (problemset if isinstance(problemset, Task) else problemset.task)
        problemset = (problemset['test'] if isinstance(problemset, Task) else problemset )
        if task.filename not in self.cache:   self.fit(task)
        if self.cache[task.filename] is None: return None  # Unsolvable mapping

        colors = self.cache[task.filename]
        output_size = task_output_grid_shape(task) # TODO: Replace with OutputGridSizeSolver().solve_grid() per problem
        outputs = [
            np.full(output_size, fill_value=color)
            for color in colors
        ]
        return outputs



if __name__ == '__main__' and not settings['production']:
    solver = XGBSingleColorSolver()
    solver.verbose = True
    filenames = [
        'training/5582e5ca.json',  # solved by SingleColorSolver
        'training/445eab21.json',  # solved by SingleColorSolver
        'training/27a28665.json',
        'training/44f52bb0.json',
        'evaluation/3194b014.json',
        'test/3194b014.json',
    ]
    for filename in filenames:
        task = Task(filename)
        plot_task(task)
        solver.plot(task)

    competition = Competition()

    for name, dataset in competition.items():
        solver.plot(dataset)

    # competition['test'].apply(solver.solve_dataset)
    competition.map(solver.solve_dataset)
    print(competition)