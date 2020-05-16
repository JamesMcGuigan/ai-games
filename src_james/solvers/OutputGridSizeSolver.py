from collections import defaultdict
from itertools import chain
from pprint import pprint
from typing import Any, Tuple

import numpy as np

from src_james.core.AbstractSolver import AbstractSolver
from src_james.core.DataModel import Competition, Task
from src_james.heuristics.Queries import Query
from src_james.solvers.old.OutputGridSolver import OutputGridSizeTransforms



class OutputGridSizeSolver(AbstractSolver):
    dtype: Tuple[int,int]
    functions = [
        OutputGridSizeTransforms.identity,
        OutputGridSizeTransforms.fixed_size,
        OutputGridSizeTransforms.ratio,
    ]
    arguments = [
        Query.grid_size_ratio_task,
        Query.count_nonzero,
        Query.unique_colors,
        1/4, 1/3, 1/2, 1, 2, 3, 4,
    ]
    def preprocess( self, input: np.ndarray ) -> Any:
        return input.shape


if __name__ == '__main__':
    solver = OutputGridSizeSolver()
    task   = Task('evaluation/68b67ca3.json')
    rule   = solver.solve(task)
    # assert rule.name == 'GridSizeIntegerMultiple'
    # assert rule.args == [(2.0, 2.0)]

    competition  = Competition()
    solutions    = defaultdict(list)
    solved_files = { name: [] for name in competition.keys() }
    error_files  = { name: [] for name in competition.keys() }
    counts       = defaultdict(int)
    for name, dataset in competition.items():
        for task in dataset:
            #rule  = solver.solve_one(task)
            rules = solver.solve(task)
            if not len(rules):
                error_files[name].append(task.filename)
            else:
                solutions[task.filename] += rules
                solved_files[name].append(task.filename)
                counts[name] += 1

    print()
    print('Counts')
    pprint(counts)

    print('Solutions')
    for filename, matching_rules in solutions.items():
        if len(matching_rules):
            print(filename)
            for rule in matching_rules: print(rule)
            print()

    print()
    print('Errors')
    for filename in chain(*error_files.values()):
        print(filename)
