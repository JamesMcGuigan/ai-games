import re
import json
import time
from itertools import chain
from typing import Dict, List

import numpy as np
import glob2

# Conceptual Mapping
# - Dataset: An array of all Tasks in the competition
# - Task:    The entire contents of a json file, outputs 1-3 lines of CSV
# - Spec:    An array of either test or training Problems
# - Problem: An input + output Grid pair
# - Grid:    An individual grid represented as a numpy arra

class Dataset:
    def __init__(self, directory: str, label: str = ''):
        self.label     = label
        self.directory = directory
        self.filenames = glob2.glob( self.directory + '/**/*.json' )
        self.tasks     = [ Task(filename) for filename in self.filenames ]

    def solve(self) -> float:
        time_start = time.perf_counter()
        for task in self.tasks:
            task.solve()
        return time.perf_counter() - time_start

    def to_csv(self):
        return "\n".join([ task.to_csv() for task in self.tasks ])

    def write_submission(self, filename='submission.csv'):
        with open(filename, 'w') as file:
            file.write(self.to_csv())


### Task:    The entire contents of a json file, outputs 1-3 lines of CSV
class Task:
    def __init__(self, filename: str):
        self.filename  = filename
        self.raw       = self.read_file(self.filename)
        self.specs     = { test_or_train: Spec(test_or_train, input_outputs, self)
                           for test_or_train, input_outputs in self.raw }

    def object_id(self, index=0) -> str:
        return re.sub('^.*/|\.json$', '', self.filename) + '_' + str(index)

    def to_csv(self) -> str:
        # TODO: We actually need to iterate over the list of potentual solutions
        return "\n".join([
            self.object_id(i) + ',' + grid.to_csv()
            for i, grid in enumerate(self.specs['test'].outputs)
        ])

    @staticmethod
    def read_file(filename: str) -> Dict[str,List[Dict[str,np.ndarray]]]:
        with open(filename, 'r') as file:
            data = json.load(file)
        for test_or_train, specs in data.items():
            for index, spec in enumerate(specs):
                for input_output, grid in spec.items():
                    data[test_or_train][index][input_output] = np.array(grid).astype('int8')
        return data

    @property
    def grids(self) -> List['Grid']:
        return list(chain(*[ spec.grids for spec in self.specs ]))


    def solve(self) -> float:
        time_start = time.perf_counter()
        raise NotImplementedError
        return time.perf_counter() - time_start



### Spec: An array of either test or training Problems
class Spec:
    def __init__(self, test_or_train: str, input_outputs: List[Dict[str,np.ndarray]], task: Task):
        self.task:          Task                       = task
        self.test_or_train: str                        = test_or_train
        self.raw:           List[Dict[str,np.ndarray]] = input_outputs
        self.problems:      List[Problem]              = [ Problem(problem, self) for problem in self.raw ]

    @property
    def inputs(self) -> List['Grid']:
        return [ problem.input for problem in self.problems ]

    @property
    def outputs(self) -> List['Grid']:
        return [ problem.output for problem in self.problems ]

    @property
    def grids(self) -> List['Grid']:
        return self.inputs + self.outputs



### Problem: An input + output Grid pair
class Problem:
    def __init__(self, problem: Dict[str,np.ndarray], spec: Spec):
        self.spec:   Spec                 = spec
        self.raw:    Dict[str,np.ndarray] = problem
        self.input:  Grid                 = Grid(problem['input'],  self)
        self.output: Grid                 = Grid(problem['output'], self)
        self.grids:  List[Grid]           = [ self.input, self.output ]

    @property
    def task(self) -> Task: return self.spec.task


### Grid: An individual grid represented as a numpy arra
class Grid:
    def __init__(self, grid: np.ndarray, problem: Problem):
        self.problem: Problem    = problem
        self.data:    np.ndarray = np.array(grid).astype('int8')

    @property
    def spec(self) -> Spec: return self.problem.spec

    @property
    def task(self) -> Task: return self.problem.spec.task

    # Source: https://www.kaggle.com/c/abstraction-and-reasoning-challenge/overview/evaluation
    def to_csv(self) -> str:
        str_pred = str([row for row in self.data.astype('int8').tolist() ])
        str_pred = str_pred.replace(', ', '')
        str_pred = str_pred.replace('[[', '|')
        str_pred = str_pred.replace('][', '|')
        str_pred = str_pred.replace(']]', '|')
        return str_pred
