#!/usr/bin/env python3

##### 
##### ./submission/kaggle_compile.py ./src_james/submission.py
##### 
##### 2020-05-15 00:59:31+01:00
##### 
##### origin	git@github.com:seshurajup/kaggle-arc.git (fetch)
##### origin	git@github.com:seshurajup/kaggle-arc.git (push)
##### 
#####   james-wip c81cf89 Solvers | work in progress - broken
##### * master    9c1db78 [ahead 5] settings.py | bugfix: settings['dir']['data'] fixed in DataModel.py
##### 
##### 9c1db78a7c884975a0d5a8d132425fdf243533a4
##### 

#####
##### START src_james/settings.py
#####

# DOCS: https://www.kaggle.com/WinningModelDocumentationGuidelines
import os
import pathlib
root_dir = pathlib.Path(__file__).parent.parent.absolute()

settings = {
    'vebose': True,
    'debug': True,
}

if os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
    settings['dir'] = {
        "data":        "../input/abstraction-and-reasoning-challenge/",
        "output":      "./",
    }
else:
    settings['dir'] = {
        "data":        os.path.join(root_dir, "./input"),
        "output":      os.path.join(root_dir, "./submission"),
    }

####################
if __name__ == '__main__':
    for dirname in settings['dir'].values():
        try:    os.makedirs(dirname, exist_ok=True)  # BUGFIX: read-only filesystem
        except: pass
    # for key,value in settings.items():  print(f"settings['{key}']:".ljust(30), str(value))


#####
##### END   src_james/settings.py
#####

#####
##### START src_james/CSV.py
#####

import os
import re
from typing import List, Union

import numpy as np

# from src_james.settings import settings



class CSV:
    @classmethod
    def write_submission(cls, dataset: 'Dataset', filename='submission.csv'):
        csv        = CSV.to_csv(dataset)
        line_count = len(csv.split('\n'))
        filename   = os.path.join(settings['dir']['output'], filename)
        with open(filename, 'w') as file:
            file.write(csv)
            print(f"\nwrote: {filename} | {line_count} lines")

    @classmethod
    def object_id(cls, filename, index=0) -> str:
        return re.sub('^.*/|\.json$', '', filename) + '_' + str(index)

    @classmethod
    def to_csv(cls, dataset: 'Dataset'):
        csv = ['output_id,output']
        for task in dataset:
            csv.append(CSV.to_csv_line(task))
        return "\n".join(csv)

    @classmethod
    def to_csv_line(cls, task: 'Task') -> str:
        csv = []
        for test_index, problem in enumerate(task['test']):
            if not problem.get('solution', None): continue
            solutions = {
                cls.grid_to_csv_string(solution)
                for solution in problem['solution']
            }
            for sol_index, solution_csv in enumerate(solutions):
                line = ",".join([
                    cls.object_id(task.filename, test_index+sol_index),
                    solution_csv
                ])
                csv.append(line)
        return "\n".join(csv)

    # Source: https://www.kaggle.com/c/abstraction-and-reasoning-challenge/overview/evaluation
    @staticmethod
    def grid_to_csv_string(grid: Union[List[List[int]], np.ndarray]) -> str:
        if isinstance(grid, np.ndarray):
            grid = grid.astype('int8').tolist()
        str_pred = str([ row for row in grid ])
        str_pred = str_pred.replace(', ', '')
        str_pred = str_pred.replace('[[', '|')
        str_pred = str_pred.replace('][', '|')
        str_pred = str_pred.replace(']]', '|')
        return str_pred


#####
##### END   src_james/CSV.py
#####

#####
##### START src_james/DataModel.py
#####

import json
import os
import re
import time
from collections import UserDict, UserList, defaultdict
from itertools import chain
from typing import Any, Dict, List, Union

import glob2
import numpy as np

# from src_james.CSV import CSV
# from src_james.settings import settings



# Conceptual Mapping
# - Competition: The collection of all Dataset in the competition
# - Dataset:     An array of all Tasks in the competition
# - Task:        The entire contents of a json file, outputs 1-3 lines of CSV
# - ProblemSet:  An array of either test or training Problems
# - Problem:     An input + output Grid pair
# - Grid:        An individual grid represented as a numpy array


class Hashed:
    __hash_counts = defaultdict(int)
    def __init__(self):
        self.__id   = f"{self.__class__.__name__}#{self.__hash_counts[self.__class__.__name__]}"
        self.__hash = hash(self.__id)
        self.__hash_counts[self.__class__.__name__] += 1
    def __hash__(self):
        return self.__hash


class Competition(UserDict, Hashed):
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

    def solve(self) -> 'Competition':
        time_start = time.perf_counter()
        for name, dataset in self.data.items():
            dataset.solve()
        self.time_taken = time.perf_counter() - time_start
        return self  # for chaining

    def score(self) -> Dict[str,Any]:
        score = { name: dataset.score() for name, dataset in self.data.items() }
        score['time'] = Dataset.to_clock(self.time_taken)
        return score

    def __str__(self):
        return "\n".join([ f"{key:11s}: {value}" for key, value in self.score().items() ])



class Dataset(UserList, Hashed):
    """ Dataset: An array of all Tasks in the competition """

    def __init__(self, directory: str, name: str = ''):
        super().__init__()
        self.name       = name
        self.directory  = directory
        self.filenames  = glob2.glob( self.directory + '/**/*.json' )
        self.filenames  = [ Task.format_filename(filename) for filename in self.filenames ]
        assert len(self.filenames), f'invalid directory: {directory}'
        self.data       = [Task(filename, self) for filename in self.filenames]
        self.time_taken = 0

    def solve(self) -> 'Dataset':
        time_start = time.perf_counter()
        for task in self:
            task.solve()
        self.time_taken = time.perf_counter() - time_start
        return self  # for chaining

    def score(self) -> Dict[str,Union[int,float]]:
        score = {}
        score['correct'] = sum([task.score() for task in self])
        score['total']   = len(self.test_outputs)
        score['error']   = round(1 - score['correct'] / score['total']) if score['total'] else 0
        score['time']    = self.to_clock(self.time_taken)
        score['name']    = self.name
        return score

    @staticmethod
    def to_clock(time_taken: float) -> str:
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


class Task(UserDict, Hashed):
    """ Task: The entire contents of a json file, outputs 1-3 lines of CSV """

    def __init__(self, filename: str, dataset: Dataset = None):
        super().__init__()

        self.filename: str = self.format_filename(filename)
        self.raw  = self.read_file( os.path.join(settings['dir']['data'], self.filename) )
        self.data = {
            test_or_train: ProblemSet(input_outputs, test_or_train, self)
            for test_or_train, input_outputs in self.raw.items()
        }

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
        return data

    @property
    def grids(self) -> List[np.ndarray]:
        return list(chain(*[ spec.grids for spec in self.data.values() ]))

    @property
    def test_outputs(self) -> List[np.ndarray]:
        return self['test'].outputs

    def solve(self) -> 'Task':
        # TODO: implement
        return self  # for chaining

    def score(self) -> int:
        return 0  # TODO: implement


class ProblemSet(UserList, Hashed):
    """ ProblemSet: An array of either test or training Problems """

    def __init__(self, input_outputs: List[Dict[str, np.ndarray]], test_or_train: str, task: Task):
        super().__init__()
        self.task:          Task                       = task
        self.test_or_train: str                        = test_or_train
        self.raw:           List[Dict[str,np.ndarray]] = input_outputs
        self.data:          List[Problem]              = [ Problem(problem, self) for problem in self.raw ]

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



class Problem(UserDict, Hashed):
    """ Problem: An input + output Grid pair """

    dtype = 'int8'
    def __init__(self, problem: Dict[str,np.ndarray], problemset: ProblemSet):
        super().__init__()
        self.problemset: ProblemSet           = problemset
        self.task:       Task                 = problemset.task
        self.raw:        Dict[str,np.ndarray] = problem
        self.data = {
            "input":    np.array(problem['input']).astype(self.dtype),
            "output":   np.array(problem['output']).astype(self.dtype) if 'output' in problem else None,
            "solution": []
        }
        self.grids:  List[np.ndarray] = [ self.data[label]
                                          for label in ['input','output']
                                          if self.data[label] is not None ]

    @property
    def filename(self): return self.task.filename


#####
##### END   src_james/DataModel.py
#####

#####
##### START ./src_james/submission.py
#####

# from src_james.DataModel import Dataset, Competition
# from src_james.settings import settings

if __name__ == '__main__':

    print('\n','-'*20,'\n')
    print('Abstraction and Reasoning Challenge')
    print('Team: Mathematicians + Experts')
    print('https://www.kaggle.com/c/abstraction-and-reasoning-challenge')
    print('\n','-'*20,'\n')

    # This is the actual competition submission entry
    # Do this first incase we have notebook timeout issues (>9h runtime)
    test_dir = f"{settings['dir']['data']}/test"
    dataset  = Dataset(test_dir, 'test')
    dataset.solve()
    for key, value in dataset.score().items():
        print( f"{key:11s}: {value}" )
    dataset.write_submission()

    print('\n','-'*20,'\n')

    # Then run the script against public competition data
    # This is mostly just to make it easier to see the stats in the published notebook logs
    competition = Competition()
    competition.solve()
    for key, value in competition.score().items():
        print( f"{key:11s}: {value}" )


#####
##### END   ./src_james/submission.py
#####

##### 
##### ./submission/kaggle_compile.py ./src_james/submission.py
##### 
##### 2020-05-15 00:59:31+01:00
##### 
##### origin	git@github.com:seshurajup/kaggle-arc.git (fetch)
##### origin	git@github.com:seshurajup/kaggle-arc.git (push)
##### 
#####   james-wip c81cf89 Solvers | work in progress - broken
##### * master    9c1db78 [ahead 5] settings.py | bugfix: settings['dir']['data'] fixed in DataModel.py
##### 
##### 9c1db78a7c884975a0d5a8d132425fdf243533a4
##### 
