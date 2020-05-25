from typing import List, Union, Callable

import numpy as np

from src_james.core.DataModel import Problem, Task, Dataset
from src_james.plot import plot_task


class Solver():
    verbose = False
    debug   = False
    def __init__(self):
        self.cache = {}

    @staticmethod
    def is_lambda_valid(_task_: Task, _function_: Callable, *args, **kwargs):  # _task_ = avoid namespace conflicts with kwargs=task
        for problem in _task_['train']:
            output = _function_(problem['input'], *args, **kwargs)
            if not np.array_equal( problem['output'], output):
                return False
        return True

    @staticmethod
    def solve_lambda( _task_: Task, _function_: Callable, *args, _inplace_=False, **kwargs) -> List[Problem]:
        solutions = []
        for index, problem in enumerate(_task_['test']):
            output = _function_(problem['input'], *args, **kwargs)
            output.flags.writeable = False
            solution = Problem({
                "input":  problem['input'],
                "output": output,
            }, problemset=_task_['test'])
            solutions.append(solution)
            if _inplace_:
                _task_['solutions'][index].append(solution)
        return solutions

    def action(self, grid: np.ndarray, task=None, *args):
        """This is the primary method this needs to be defined"""
        return grid
        # raise NotImplementedError()

    def detect(self, task: Task) -> bool:
        """default heuristic is simply to run the solver"""
        return self.test(task)

    def fit(self, task: Task):
        if task.filename in self.cache: return
        pass

    def test(self, task: Task) -> bool:
        """test if the given action correctly solves the task"""
        if task.filename not in self.cache: self.fit(task)

        args = self.cache.get(task.filename, ())
        return self.is_lambda_valid(task, self.action, *args, task=task)

    def solve(self, task: Task, force=False, inplace=True) -> Union[List[Problem],None]:
        """solve test case and persist"""
        if task.filename not in self.cache: self.fit(task)
        try:
            if self.detect(task) or force:    # may generate cache
                if self.test(task) or force:  # may generate cache
                    args     = self.cache.get(task.filename, ())
                    if self.verbose and args:
                        print('solved: ', self.__class__.__name__, args)
                    if isinstance(args, dict):
                        solutions = self.solve_lambda(task, self.action, **args, task=task, _inplace_=True)
                    else:
                        solutions = self.solve_lambda(task, self.action,  *args, task=task, _inplace_=True )
                    return solutions
        except Exception as exception:
            if self.debug: raise exception
        return None

    def solve_all(self, tasks: Union[Dataset,List[Task]], plot=False, solve_detects=False):
        count = 0
        for task in tasks:
            if self.detect(task):
                solution = self.solve(task, force=solve_detects)
                if solution or (solve_detects and self.test(task)):
                    count += 1
                    # solution = self.solve(task, force=solve_detects)
                    if plot == True:
                        plot_task(task)
        return count

    def plot(self, tasks: Union[Dataset,List[Task], Task]):
        if isinstance(tasks, Task): tasks = [ tasks ]
        return self.solve_all(tasks, plot=True, solve_detects=False)

    def plot_detects(self, tasks: Union[Dataset,List[Task],Task], unsolved=True):
        if isinstance(tasks, Task): tasks = [ tasks ]
        if unsolved:
            tasks = [ task for task in tasks if not task.solutions_count ]
        return self.solve_all(tasks, plot=True, solve_detects=True)
