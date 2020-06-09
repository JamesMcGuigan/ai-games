from collections import UserDict
from typing import Any
from typing import Dict

import time

from src.datamodel.Dataset import Dataset
from src.settings import settings


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


