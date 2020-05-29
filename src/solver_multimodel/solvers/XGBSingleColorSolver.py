from typing import List
from typing import Union

from src.datamodel.Competition import Competition
from src.datamodel.ProblemSet import ProblemSet
from src.datamodel.Task import Task
from src.settings import settings
from src.solver_multimodel.core.ProblemSetSolver import ProblemSetSolver
from src.solver_multimodel.functions.queries.colors import task_is_singlecolor
from src.solver_multimodel.functions.queries.grid import *
from src.solver_multimodel.functions.queries.ratio import is_task_output_grid_shape_constant
from src.solver_multimodel.functions.queries.ratio import task_output_grid_shape
from src.solver_multimodel.solvers.XGBSingleColorEncoder import SingleColorXGBEncoder
from src.util.plot import plot_task


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