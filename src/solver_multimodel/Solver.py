from collections import UserList
from typing import Callable
from typing import List
from typing import Union

import numpy as np

from src.core.DataModel import Dataset
from src.core.DataModel import Problem
from src.core.DataModel import Task
from src.plot import plot_task


class Solver():
    verbose = False
    debug   = False
    def __init__(self):
        self.cache = {}


    def detect(self, task: Task) -> bool:
        """ @override | default heuristic is simply to run the solver"""
        return self.test(task)


    def fit(self, task: Task):
        """ @override | sets: self.cache[task.filename] """
        if task.filename in self.cache: return
        pass


    def solve_grid(self, grid: np.ndarray, *args, task=None, **kwargs):
        """ @override | This is the primary method this needs to be defined"""
        raise NotImplementedError
        # return grid
        # raise NotImplementedError()


    def test(self, task: Task) -> bool:
        """test if the given .solve_grid() correctly solves the task"""
        if task.filename not in self.cache: self.fit(task)
        if self.cache.get(task.filename, True) is None: return False

        args = self.cache.get(task.filename, ())
        return self.is_lambda_valid(task, self.solve_grid, *args, task=task)


    def is_lambda_valid(self, _task_: Task, _function_: Callable, *args, **kwargs):  # _task_ = avoid namespace conflicts with kwargs=task
        for problem in _task_['train']:
            output = _function_(problem['input'], *args, **kwargs)
            if not np.array_equal( problem['output'], output):
                return False
        return True


    def format_args(self, args):
        if isinstance(args, dict):
            args = dict(zip(args.keys(), map(self.format_args, list(args.values()))))
        elif isinstance(args, (list,set,tuple)):
            args = list(args)
            for index, arg in enumerate(args):
                if hasattr(arg, '__name__'):
                    arg = f"<{type(arg).__name__}:{arg.__name__}>"
                if isinstance(arg, (list,set,tuple,dict)):
                    arg = self.format_args(arg)
                args[index] = arg
            args = tuple(args)
        return args


    def log_solved(self, task: Task, args: Union[list,tuple,set], solutions: List[Problem]):
        if self.verbose:
            if 'test' in task.filename:           label = 'test  '
            elif self.is_solved(task, solutions): label = 'solved'
            else:                                 label = 'guess '

            args  = self.format_args(args) if len(args) else None
            print(f'{label}:', task.filename, self.__class__.__name__, args)


    def is_solved(self, task: Task, solutions: List[Problem]):
        for solution in solutions:
            for problem in task['test']:
                if solution == problem:
                    return True
        return False


    def solve(self, task: Task, force=False, inplace=True) -> Union[List[Problem],None]:
        """solve test case and persist"""
        if task.filename not in self.cache:             self.fit(task)
        if self.cache.get(task.filename, True) is None: return None
        try:
            if self.detect(task) or force:    # may generate cache
                if self.test(task) or force:  # may generate cache
                    args = self.cache.get(task.filename, ())
                    if args is None: return None
                    if isinstance(args, dict):
                        solutions = self.solve_task(task, self.solve_grid, **args, task=task)
                    else:
                        solutions = self.solve_task(task, self.solve_grid, *args, task=task)

                    for index, solution in enumerate(solutions):
                        task['solutions'][index].append(solution)
                    if len(solutions):
                        self.log_solved(task, args, solutions)
                    return solutions
        except Exception as exception:
            if self.debug: raise exception
        return None


    def solve_dataset(self, tasks: Union[Dataset, List[Task]], plot=False, solve_detects=False):
        count = 0
        for task in tasks:
            if self.detect(task):
                solution = self.solve(task, force=solve_detects)
                if solution or (solve_detects and self.test(task)):
                    count += 1
                    if plot == True:
                        plot_task(task)
        return count


    def solve_task(self, _task_: Task, _function_: Callable, *args, _inplace_=False, **kwargs) -> List[Problem]:
        solutions = []
        for index, problem in enumerate(_task_['test']):
            solution = self.solve_problem(
                _problem_  = problem,
                _task_     = _task_,
                _function_ = _function_,
                *args,
                **kwargs
            )
            solutions.append(solution)
        return solutions


    def solve_problem(self, *args, _problem_: Problem, _task_: Task, _function_: Callable, **kwargs) -> Problem:
        output = _function_(_problem_['input'], *args, **kwargs)
        solution = Problem({
            "input":  _problem_['input'],
            "output": output,
        }, problemset=_task_['test'])
        return solution


    def plot(self, tasks: Union[Dataset,List[Task], Task]):
        if not isinstance(tasks, (list,UserList)): tasks = [ tasks ]
        return self.solve_dataset(tasks, plot=True, solve_detects=False)


    def plot_detects(self, tasks: Union[Dataset,List[Task],Task], unsolved=True):
        if not isinstance(tasks, (list,UserList)): tasks = [ tasks ]
        if unsolved:
            tasks = [ task for task in tasks if not task.solutions_count ]
        return self.solve_dataset(tasks, plot=True, solve_detects=True)
