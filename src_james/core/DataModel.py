import json
import os
import re
import time
from collections import UserDict
from collections import UserList
from itertools import chain
from typing import Any
from typing import Dict
from typing import List
from typing import Union

import glob2
import numpy as np

from src_james.core.CSV import CSV
from src_james.settings import settings


# Conceptual Mapping
# - Competition: The collection of all Dataset in the competition
# - Dataset:     An array of all Tasks in the competition
# - Task:        The entire contents of a json file, outputs 1-3 lines of CSV
# - ProblemSet:  An array of either test or training Problems
# - Problem:     An input + output Grid pair
# - Grid:        An individual grid represented as a numpy array


class Competition(UserDict):
    """ Competition: The collection of all Dataset in the competition """

    def __init__(self):
        super().__init__()
        self.time_taken  = 0
        self.directories = {
            name: f"{settings['dir']['data']}/{name}"
            for name in ['training', 'evaluation', 'test']
        }
        self.data = {
            name: Dataset(directory, name)
            for name, directory in self.directories.items()
        }

    def __str__(self):
        return "\n".join([ f"{key:11s}: {value}" for key, value in self.score().items() ])

    def solve(self) -> 'Competition':
        time_start = time.perf_counter()
        for name, dataset in self.data.items():
            dataset.solve()
        self.time_taken = time.perf_counter() - time_start
        return self  # for chaining

    def score(self) -> Dict[str,Any]:
        score = { name: dataset.score() for name, dataset in self.data.items() }
        success_ratio = score['evaluation']['correct'] / max(1e-10, score['evaluation']['guesses'])
        score['test']['correct'] = round(score['test']['guesses'] * success_ratio, 1)
        score['time'] = Dataset.format_clock(self.time_taken)
        return score

    @classmethod
    def format_clock(cls, time_taken: float) -> str:
        return Dataset.format_clock(time_taken)

    def map(self, function):
        output = []
        competition = self
        competition.time_start = time.perf_counter()
        for name, dataset in competition.items():
            result = dataset.apply(function)
            output.append( result )
        competition.time_taken = time.perf_counter() - competition.time_start
        return output



class Dataset(UserList):
    """ Dataset: An array of all Tasks in the competition """

    def __init__(self, directory: str, name: str = ''):
        super().__init__()
        self.name       = name
        self.directory  = directory
        self.filenames  = glob2.glob( self.directory + '/**/*.json' )
        self.filenames  = sorted([ Task.format_filename(filename) for filename in self.filenames ])
        assert len(self.filenames), f'invalid directory: {directory}'
        self.data       = [Task(filename, self) for filename in self.filenames]
        self.time_taken = 0

    def __repr__(self):
        return f'<{self.__class__.__name__}:{self.directory}>'

    def __hash__(self):
        return hash(self.directory)

    def __eq__(self, other):
        if not isinstance(other, Dataset): return False
        return self.directory == other.directory

    def apply(self, function):
        dataset = self
        dataset.time_start = time.perf_counter()
        result = function(dataset)
        dataset.time_taken = time.perf_counter() - dataset.time_start
        return result

    def solve(self) -> 'Dataset':
        time_start = time.perf_counter()
        for task in self:
            task.solve()
        self.time_taken = time.perf_counter() - time_start
        return self  # for chaining

    def score(self) -> Dict[str,Union[int,float]]:
        score = {}
        score['correct'] = sum([task.score()   for task in self])
        score['guesses'] = sum([task.guesses() for task in self])
        score['total']   = len(self.test_outputs)
        score['error']   = round(1 - score['correct'] / score['total'],4) if score['total'] else 0
        score['time']    = self.format_clock(self.time_taken)
        score['name']    = self.name
        return score

    @classmethod
    def format_clock(cls, time_taken: float) -> str:
        hours   = time_taken // (60 * 60)
        minutes = time_taken // (60)
        seconds = time_taken % 60
        clock   = "{:02.0f}:{:02.0f}:{:02.0f}".format(hours,minutes,seconds)
        return clock

    def to_csv(self):
        return CSV.to_csv(self)

    def write_submission(self, filename='submission.csv'):
        return CSV.write_submission(self, filename)

    @property
    def test_outputs(self) -> List[np.ndarray]:
        return list(chain(*[task.test_outputs for task in self.data]))


class Task(UserDict):
    """ Task: The entire contents of a json file, outputs 1-3 lines of CSV """

    def __init__(self, filename: str, dataset: Dataset = None):
        super().__init__()

        self.dataset: Dataset = dataset
        self.filename: str    = self.format_filename(filename)
        self.raw  = self.read_file( os.path.join(settings['dir']['data'], self.filename) )
        self.data = {
            test_or_train: ProblemSet(input_outputs, test_or_train, self)
            for test_or_train, input_outputs in self.raw.items()
        }
        self.data['solutions']: List[ProblemSet] = [
            ProblemSet([], test_or_train='solutions', task=self) for task in self.data['test']
        ]

    def __repr__(self):
        return f'<{self.__class__.__name__}:{self.filename}>'

    def __hash__(self):
        return hash(self.filename)

    @classmethod
    def format_filename(cls, filename):
        return re.sub(r'^(.*/)?(\w+/\w+\.json)$', r'\2', filename)

    @staticmethod
    def read_file(filename: str) -> Dict[str,List[Dict[str,np.ndarray]]]:
        with open(filename, 'r') as file:
            data = json.load(file)
        for test_or_train, specs in data.items():
            for index, spec in enumerate(specs):
                for input_output, grid in spec.items():
                    data[test_or_train][index][input_output] = np.array(grid).astype('int8')
                    data[test_or_train][index][input_output].flags.writeable = False  # make immutable
        return data

    @property
    def grids(self) -> List[np.ndarray]:
        return list(chain(*[ spec.grids for spec in self.data.values() ]))

    @property
    def test_outputs(self) -> List[np.ndarray]:
        return self['test'].outputs

    def solve(self) -> 'Task':
        # TODO: implement
        print(self.__class__.__name__, 'solve()', NotImplementedError())
        return self  # for chaining

    def make_solutions_unique(self):
        self.data['solutions'] = [
            problemset.unique() for problemset in self.data['solutions']
        ]


    @property
    def is_solved(self):
        return all(map(len, self.data['solutions']))

    @property
    def solutions_count(self):
        return sum(map(len, self.data['solutions']))

    def score(self) -> int:
        score = 0
        # self.make_solutions_unique()  # Is causing exceptions
        for index, test_problem in enumerate(self.data['test']):
            for solution in self.data['solutions'][index]:
                if test_problem == solution:
                    score += 1
                    break
        return min(score, self.max_score())

    def guesses(self) -> int:
        score = 0
        for index, test_problem in enumerate(self.data['test']):
            if len(self.data['solutions'][index]):
                score += 1
        return min(score, self.max_score())

    def max_score(self) -> int:
        return len(self.data['test'])


class ProblemSet(UserList):
    """ ProblemSet: An array of either test or training Problems """
    _instance_count = 0
    def __init__(self, input_outputs: List[Dict[str, np.ndarray]], test_or_train: str, task: Task):
        super().__init__()
        self.task:          Task                       = task
        self.test_or_train: str                        = test_or_train
        self.raw:           List[Dict[str,np.ndarray]] = input_outputs
        self.data:          List[Problem]              = [ Problem(problem, self) for problem in self.raw ]
        self._id = self.__class__._instance_count = self.__class__._instance_count + 1

    def __eq__(self, other):
        if not isinstance(other, ProblemSet): return False
        return self._id == other._id

    def __hash__(self):
        return self._id

    def unique(self) -> 'ProblemSet':
        # unique = list({ hash(problem): problem for problem in self.data }.values())
        unique = set( problem for problem in self.data )
        if len(self.data) == len(unique):
            return self
        else:
            unique = [ problem.raw for problem in self.data ]
            return ProblemSet(unique, test_or_train=self.test_or_train, task=self.task)

    @property
    def filename(self): return self.task.filename

    @property
    def inputs(self) -> List[np.ndarray]:
        return [ problem['input'] for problem in self.data if problem ]

    @property
    def outputs(self) -> List[np.ndarray]:
        return [ problem['output'] for problem in self.data if problem ]

    @property
    def grids(self) -> List[np.ndarray]:
        return self.inputs + self.outputs

class Problem(UserDict):
    """ Problem: An input + output Grid pair """
    dtype = 'int8'
    def __init__(self, problem: Dict[str,np.ndarray], problemset: ProblemSet):
        super().__init__()
        self._hash = 0
        self.problemset: ProblemSet           = problemset
        self.task:       Task                 = problemset.task
        self.raw:        Dict[str,np.ndarray] = problem

        self.data = {}
        for key in ['input', 'output']:
            value = self.cast(problem.get(key, None))
            self.data[key] = value

    def cast(self, value: Any):
        if value is None: return None
        # value = np.ascontiguousarray(value, dtype=self.dtype)  # disable: could potntually mess with hashing
        value.flags.writeable = False
        return value

    @property
    def grids(self) -> List[np.ndarray]:
        return  [ self.data[label]
                  for label in ['input','output']
                  if self.data[label] is not None ]

    @property
    def filename(self): return self.task.filename

    def __eq__(self, other):
        if not isinstance(other, (Problem, dict, UserDict)): return False
        for label in ['input','output']:
            if label in self  and label not in other: return False
            if label in other and label not in self:  return False
            if not np.array_equal(self[label], other[label]):
                return False
        return True

    def __hash__(self):
        if not self._hash:
            for item in [ self.data['input'], self.data['output'] ]:
                item = item.tobytes() if isinstance(item, np.ndarray) else item
                self._hash += hash(item)
        return self._hash

