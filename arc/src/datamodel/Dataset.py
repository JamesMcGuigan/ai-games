from collections import UserList
from typing import Dict
from typing import List
from typing import Union

import glob2
import numpy as np
import time
from itertools import chain

from src.datamodel.CSV import CSV
from src.datamodel.Task import Task


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
        minutes = time_taken // 60
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
