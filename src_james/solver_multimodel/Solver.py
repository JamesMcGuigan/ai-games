from copy import deepcopy
from itertools import chain
from typing import List

import numpy as np

from src_james.core.DataModel import Problem
from src_james.plot import plot_task


class Solver():
    verbose = False
    debug   = False
    def __init__(self):
        self.cache = {}

    @staticmethod
    def loop_specs(task, test_train='train'):
        specs = task[test_train]
        for index, spec in enumerate(specs):
            yield { 'input': spec['input'], 'output': spec['output'] }

    @staticmethod
    def is_lambda_valid(_task_, _function_, *args, **kwargs):  # _task_ = avoid namespace conflicts with kwargs=task
        for spec in Solver.loop_specs(_task_, 'train'):
            output = _function_(spec['input'], *args, **kwargs)
            if not np.array_equal( spec['output'], output):
                return False
        return True

    @staticmethod
    def solve_lambda(_task_, _function_, *args, _inplace_=False, **kwargs) -> List[Problem]:
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
            _task_['solutions'] += solutions
        return solutions

    def action(self, grid, task=None, *args):
        """This is the primary method this needs to be defined"""
        return grid
        # raise NotImplementedError()

    def detect(self, task):
        """default heuristic is simply to run the solver"""
        return self.test(task)

    def test(self, task):
        """test if the given action correctly solves the task"""
        args = self.cache.get(task.filename, ())
        return self.is_lambda_valid(task, self.action, *args, task=task)

    def solve(self, task, force=False, inplace=True):
        """solve test case and persist"""
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

    def solve_all(self, tasks, plot=False, solve_detects=False):
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

    def plot(self, tasks):
        return self.solve_all(tasks, plot=True, solve_detects=False)

    def plot_detects(self, tasks, unsolved=True):
        if unsolved:
            tasks = { file: task for (file,task) in deepcopy(tasks).items() if not 'solution' in task }
        return self.solve_all(tasks, plot=True, solve_detects=True)


    ### Helper Methods ###

    @staticmethod
    def grid_shape_ratio(grid1, grid2):
        try:
            return ( grid2.shape[0] / grid1.shape[0], grid2.shape[1] / grid1.shape[1] )
        except:
            return (0, 0)  # For tests

    @staticmethod
    def task_grids(task):
        grids = []
        for test_train in ['test','train']:
            for spec in task[test_train]:
                grids += [ spec.get('input',[]), spec.get('output',[]) ]  # tests not gaurenteed to have outputs
        return grids

    @staticmethod
    def task_grid_shapes(task):
        return [ np.array(grid).shape for grid in Solver.task_grids(task) ]

    @staticmethod
    def task_grid_max_dim(task):
        return max(chain(*Solver.task_grid_shapes(task)))

    @staticmethod
    def task_shape_ratios(task):
        ratios = set([
            Solver.grid_shape_ratio(spec.get('input',[]), spec.get('output',[]))
            for spec in Solver.loop_specs(task, 'train')
        ])
        # ratios = set([ int(ratio) if ratio.is_integer() else ratio for ratio in chain(*ratios) ])
        return ratios

    @staticmethod
    def is_task_shape_ratio_unchanged(task):
        return Solver.task_shape_ratios(task) == { (1,1) }

    @staticmethod
    def is_task_shape_ratio_consistant(task):
        return len(Solver.task_shape_ratios(task)) == 1

    @staticmethod
    def is_task_shape_ratio_integer_multiple(task):
        ratios = Solver.task_shape_ratios(task)
        return all([ isinstance(d, int) or d.is_integer() for d in chain(*ratios) ])

