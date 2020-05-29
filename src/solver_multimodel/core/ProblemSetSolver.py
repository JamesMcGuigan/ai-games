from collections import Callable
from typing import List
from typing import Union

import numpy as np

from src.datamodel.Problem import Problem
from src.datamodel.ProblemSet import ProblemSet
from src.datamodel.Task import Task
from src.solver_multimodel.core.Solver import Solver


class ProblemSetSolver(Solver):
    def solve_task(self, _task_: Task, _function_: Callable, *args, _inplace_=False, **kwargs) -> List[Problem]:
        self.fit(_task_)
        if not _task_.filename in self.cache:   return []
        if self.cache[_task_.filename] is None: return []

        solutions = self.solve_grid(_task_['test'])
        if solutions is None: return []

        problemset = self.cast_problemset(solutions, task=_task_)
        problems   = list(problemset)
        return problems


    def cast_problems(self, solutions: List[np.ndarray], task: Task) -> List[Problem]:
        problemset = task['test']
        problems   = []
        for index, solution in enumerate(solutions):
            problem = Problem({
                "input":  problemset[index]['input'],
                "output": solution,
            }, problemset=problemset)
            problems.append(problem)
        return problems


    def cast_problemset(self, solutions: List[np.ndarray], task: Task) -> ProblemSet:
        problems = self.cast_problems(solutions, task=task)
        output   = ProblemSet(problems, task=task, test_or_train='solutions')
        return output


    def predict(self, problemset: Union[ProblemSet,Task], *args, task: Task=None, **kwargs) -> Union[None,List[np.ndarray]]:
        task       = task or (problemset if isinstance(problemset, Task) else problemset.task)
        problemset = (problemset['test'] if isinstance(problemset, Task) else problemset )
        if task.filename not in self.cache:   self.fit(task)
        if self.cache[task.filename] is None: return None  # Unsolvable mapping
        raise NotImplementedError


    def test(self, task: Task) -> bool:
        """test if .predict() correctly solves the task"""
        self.fit(task)
        if not task.filename in self.cache:   return False
        if self.cache[task.filename] is None: return False

        problemset = task['train']
        training_predictions = self.predict(problemset, task=task)
        tests_pass = bool( len(training_predictions) == len(problemset) )
        for index, prediction in enumerate(training_predictions):
            if not tests_pass: break
            if not np.array_equal( task['train'][index]['output'], prediction ):
                tests_pass = False
        return tests_pass

