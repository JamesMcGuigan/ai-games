import json
import os
import re
from collections import UserDict
from typing import Dict
from typing import List

import numpy as np
from itertools import chain

from src.datamodel.ProblemSet import ProblemSet
from src.settings import settings


# noinspection PyUnresolvedReferences
class Task(UserDict):
    """ Task: The entire contents of a json file, outputs 1-3 lines of CSV """

    def __init__(self, filename: str, dataset: 'Dataset' = None):
        super().__init__()

        self.dataset: 'Dataset' = dataset
        self.filename: str      = self.format_filename(filename)
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

