#!/usr/bin/env python3

##### 
##### ./kaggle_compile.py ./src_james/submission.py --save
##### 
##### 2020-05-08 23:04:44+01:00
##### 
##### origin	git@github.com:seshurajup/kaggle-arc.git (fetch)
##### origin	git@github.com:seshurajup/kaggle-arc.git (push)
##### 
##### * master 170f81c [ahead 1] DataModel | implement submission.py + submission.csv pipeline
##### 
##### 170f81c444f3c2470473cd92cd8072d8fbec4d7e
##### 
##### Wrote: ./submission.py

#####
##### START src_james/settings.py
#####

# DOCS: https://www.kaggle.com/WinningModelDocumentationGuidelines
import os

settings = {}

if os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
    settings['dir'] = {
        "data":        "../input/abstraction-and-reasoning-challenge/",
        "logs":        "./logs",
    }
else:
    settings['dir'] = {
        "data":        "./input",
        "logs":        "./logs",
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
##### START src_james/DataModel.py
#####

import json
import re
import time
from collections import UserDict, UserList
from itertools import chain
from typing import Dict, List, Union, Any

import glob2
import numpy as np

# from src_james.settings import settings


# Conceptual Mapping
# - Competition: The collection of all Dataset in the competition
# - Dataset: An array of all Tasks in the competition
# - Task:    The entire contents of a json file, outputs 1-3 lines of CSV
# - Spec:    An array of either test or training Problems
# - Problem: An input + output Grid pair
# - Grid:    An individual grid represented as a numpy arra


### Competition: The collection of all Dataset in the competition
class Competition:
    def __init__(self):
        self.directories = {
            name: f"{settings['dir']['data']}/{name}"
            for name in ['training', 'evaluation', 'test']
        }
        self.datasets = {
            name: Dataset(directory, name)
            for name, directory in self.directories.items()
        }

    def solve(self) -> 'Competition':
        time_start = time.perf_counter()
        for name, dataset in self.datasets.items():
            dataset.solve()
        self.time_taken = time.perf_counter() - time_start
        return self  # for chaining

    def score(self) -> Dict[str,Any]:
        score = { name: dataset.score() for name, dataset in self.datasets.items() }
        score['time'] = Dataset.to_clock(self.time_taken)
        return score

    def __str__(self):
        return "\n".join([ f"{key:11s}: {value}" for key, value in self.score().items() ])



class Dataset(UserList):
    def __init__(self, directory: str, name: str = ''):
        super().__init__()
        self.name       = name
        self.directory  = directory
        self.filenames  = glob2.glob( self.directory + '/**/*.json' )
        assert len(self.filenames), f'invalid directory: {directory}'
        self.data       = [Task(filename) for filename in self.filenames]
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
        csv = ['output_id,output']
        for task in self:
            csv.append(task.to_csv_line())
        return "\n".join(csv)

    def write_submission(self, filename='submission.csv'):
        csv        = self.to_csv()
        line_count = len(csv.split('\n'))
        with open(filename, 'w') as file:
            file.write(csv)
            print(f"\nwrote: {filename} | {line_count} lines")

    @property
    def test_outputs(self) -> List['Grid']:
        return list(chain(*[task.test_outputs for task in self.data]))


### Task:    The entire contents of a json file, outputs 1-3 lines of CSV
class Task(UserDict):
    def __init__(self, filename: str, **kwargs):
        super().__init__(**kwargs)
        self.filename  = filename
        self.raw       = self.read_file(self.filename)
        self.data = {
            test_or_train: Spec(test_or_train, input_outputs, self)
            for test_or_train, input_outputs in self.raw.items()
        }

    def object_id(self, index=0) -> str:
        return re.sub('^.*/|\.json$', '', self.filename) + '_' + str(index)

    def to_csv_line(self) -> str:
        # TODO: We actually need to iterate over the list of potential solutions
        csv = []
        for i, problem in enumerate(self['test']):
            csv.append(self.object_id(i) + ',' + problem.to_csv_string())
        return "\n".join(csv)

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
        return list(chain(*[ spec.grids for spec in self.values() ]))

    @property
    def test_outputs(self) -> List['Grid']:
        return self['test'].outputs

    def solve(self) -> 'Task':
        # TODO: implement
        return self  # for chaining

    def score(self) -> int:
        return 0  # TODO: implement


### Spec: An array of either test or training Problems
class Spec(UserList):
    def __init__(self, test_or_train: str, input_outputs: List[Dict[str, np.ndarray]], task: Task):
        super().__init__()
        self.task:          Task                       = task
        self.test_or_train: str                        = test_or_train
        self.raw:           List[Dict[str,np.ndarray]] = input_outputs
        self.data:          List[Problem]              = [ Problem(problem, self) for problem in self.raw ]

    @property
    def inputs(self) -> List['Grid']:
        return [ problem['input'] for problem in self if problem ]

    @property
    def outputs(self) -> List['Grid']:
        return [ problem['output'] for problem in self if problem ]

    @property
    def grids(self) -> List['Grid']:
        return self.inputs + self.outputs


### Problem: An input + output Grid pair
class Problem(UserDict):
    def __init__(self, problem: Dict[str, np.ndarray], spec: Spec, **kwargs):
        super().__init__(**kwargs)
        self.spec:     Spec                 = spec
        self.raw:      Dict[str,np.ndarray] = problem
        self.data = {
            "input":    Grid(problem['input'],  self),
            "output":   Grid(problem['output'], self) if 'output' in problem else None,
            "solution": None
        }
        self.grids:  List[Grid] = [ grid for grid in [self['input'], self['output']] if grid ]

    @property
    def task(self) -> Task: return self.spec.task

    def to_csv_string(self) -> str:
        # TODO: Do we need to consider a range of possible solutions?
        if self['solution']: return self['solution'].to_csv_string()
        else:                return self['input'].to_csv_string()


### Grid: An individual grid represented as a numpy arra
class Grid():
    def __init__(self, grid: np.ndarray, problem: Problem):
        super().__init__()
        self.problem: Problem        = problem
        self.data:    np.ndarray     = np.array(grid).astype('int8')

    def __getattr__(self, attr):
        return getattr(self.data, attr)

    @property
    def spec(self) -> Spec: return self.problem.spec

    @property
    def task(self) -> Task: return self.problem.spec.task

    # Source: https://www.kaggle.com/c/abstraction-and-reasoning-challenge/overview/evaluation
    def to_csv_string(self) -> str:
        # noinspection PyTypeChecker
        str_pred = str([ row for row in self.data.astype('int8').tolist() ])
        str_pred = str_pred.replace(', ', '')
        str_pred = str_pred.replace('[[', '|')
        str_pred = str_pred.replace('][', '|')
        str_pred = str_pred.replace(']]', '|')
        return str_pred


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
##### ./kaggle_compile.py ./src_james/submission.py --save
##### 
##### 2020-05-08 23:04:44+01:00
##### 
##### origin	git@github.com:seshurajup/kaggle-arc.git (fetch)
##### origin	git@github.com:seshurajup/kaggle-arc.git (push)
##### 
##### * master 170f81c [ahead 1] DataModel | implement submission.py + submission.csv pipeline
##### 
##### 170f81c444f3c2470473cd92cd8072d8fbec4d7e
##### 
##### Wrote: ./submission.py