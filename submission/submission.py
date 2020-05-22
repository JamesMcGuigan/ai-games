#!/usr/bin/env python3

##### 
##### ./submission/kaggle_compile.py ./src_james/original/main.py
##### 
##### 2020-05-22 22:41:08+01:00
##### 
##### origin	git@github.com:seshurajup/kaggle-arc.git (fetch)
##### origin	git@github.com:seshurajup/kaggle-arc.git (push)
##### 
#####   james-wip c81cf89 Solvers | work in progress - broken
##### * master    08f4dab import original notebook codebase
##### 
##### 08f4dabd9a073fb1d962168fed1d5751eae5572a
##### 

#####
##### START src_james/settings.py
#####

# DOCS: https://www.kaggle.com/WinningModelDocumentationGuidelines
import os
import pathlib
root_dir = pathlib.Path(__file__).parent.parent.absolute()

settings = {
    'verbose': True,
    'debug':   not os.environ.get('KAGGLE_KERNEL_RUN_TYPE'),
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
##### START src_james/core/CSV.py
#####

import os
import re

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
            line = CSV.to_csv_line(task)
            if line: csv.append(line)
        return "\n".join(csv)

    @classmethod
    def to_csv_line(cls, task: 'Task') -> str:
        csv = []
        solutions = set({
            cls.grid_to_csv_string(problem['output'])
            for problem in task['solutions']
        })
        for index, solution_csv in enumerate(solutions):
            if not solution_csv: continue
            line = ",".join([
                cls.object_id(task.filename, index),
                solution_csv
            ])
            csv.append(line)
        return "\n".join(csv)

    # Source: https://www.kaggle.com/c/abstraction-and-reasoning-challenge/overview/evaluation
    # noinspection PyTypeChecker
    @staticmethod
    def grid_to_csv_string(grid: np.ndarray) -> str:
        if grid is None: return None
        grid = np.array(grid).astype('int8').tolist()
        str_pred = str([ row for row in grid ])
        str_pred = str_pred.replace(', ', '')
        str_pred = str_pred.replace('[[', '|')
        str_pred = str_pred.replace('][', '|')
        str_pred = str_pred.replace(']]', '|')
        return str_pred


#####
##### END   src_james/core/CSV.py
#####

#####
##### START src_james/core/DataModel.py
#####

import json
import os
import re
import time
from collections import UserDict, UserList
from itertools import chain
from typing import Any, Dict, List, Union

import glob2
import numpy as np

# from src_james.core.CSV import CSV
# from src_james.settings import settings



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

    def solve(self) -> 'Competition':
        time_start = time.perf_counter()
        for name, dataset in self.data.items():
            dataset.solve()
        self.time_taken = time.perf_counter() - time_start
        return self  # for chaining

    def score(self) -> Dict[str,Any]:
        score = { name: dataset.score() for name, dataset in self.data.items() }
        score['time'] = Dataset.format_clock(self.time_taken)
        return score

    @classmethod
    def format_clock(cls, time_taken: float) -> str:
        return Dataset.format_clock(time_taken)

    def __str__(self):
        return "\n".join([ f"{key:11s}: {value}" for key, value in self.score().items() ])



class Dataset(UserList):
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

    def __repr__(self):
        return f'<{self.__class__.__name__}:{self.directory}>'

    def __hash__(self):
        return hash(self.directory)

    def __eq__(self, other):
        if not isinstance(other, Dataset): return False
        return self.directory == other.directory


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

        self.filename: str = self.format_filename(filename)
        self.raw  = self.read_file( os.path.join(settings['dir']['data'], self.filename) )
        self.data = {
            test_or_train: ProblemSet(input_outputs, test_or_train, self)
            for test_or_train, input_outputs in self.raw.items()
        }
        self.data['solutions'] = []

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

    def score(self) -> int:
        score = len(set([
            problem['output'].tobytes()
            for problem in self.data['solutions']
            if problem['output'] is not None
        ]))
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
        if not isinstance(ProblemSet, other): return False
        return self._id == other._id

    def __hash__(self):
        return self._id

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
        self.data = {
            "input":    np.array(problem['input']).astype(self.dtype),
            "output":   np.array(problem['output']).astype(self.dtype) if 'output' in problem else None,
        }
        self.grids:  List[np.ndarray] = [ self.data[label]
                                          for label in ['input','output']
                                          if self.data[label] is not None ]

    @property
    def filename(self): return self.task.filename

    def __hash__(self):
        if not self._hash:
            for item in [ self.data['input'], self.data['output'] ]:
                item = item.tobytes() if isinstance(item, (np.ndarray, np.generic)) else item
                self._hash += hash(item)
        return self._hash



#####
##### END   src_james/core/DataModel.py
#####

#####
##### START src_james/plot.py
#####

# Source: https://www.kaggle.com/jamesmcguigan/arc-geometry-solvers/

import matplotlib.pyplot as plt
from matplotlib import colors

# Modified from: https://www.kaggle.com/zaharch/visualizing-all-tasks-updated
# from src_james.core.DataModel import Task



def plot_one(task, ax, i,train_or_test,input_or_output):
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)

    try:
        input_matrix = task[train_or_test][i][input_or_output]
        ax.imshow(input_matrix, cmap=cmap, norm=norm)
        ax.grid(True,which='both',color='lightgrey', linewidth=0.5)
        ax.set_yticks([x-0.5 for x in range(1+len(input_matrix))])
        ax.set_xticks([x-0.5 for x in range(1+len(input_matrix[0]))])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(train_or_test + ' '+input_or_output)
    except: pass  # mat throw on tests, as they have not "output"

def plot_task(task: Task, scale=2):
    """
    Plots the first train and test pairs of a specified task,
    using same color scheme as the ARC app
    """
    if isinstance(task, str): task = Task(task)
    filename = task.filename

    num_train = len(task['train']) + len(task['test']) + 1
    if len(task['solutions']): num_train += len(task['solutions']) + 1

    fig, axs = plt.subplots(2, num_train, figsize=(scale*num_train,scale*2))
    if filename: fig.suptitle(filename)

    i = 0
    for i in range(len(task['train'])):
        plot_one(task, axs[0,i],i,'train','input')
        plot_one(task, axs[1,i],i,'train','output')

    axs[0,i+1].axis('off'); axs[1,i+1].axis('off')
    j = 0
    for j in range(len(task['test'])):
        plot_one(task, axs[0,i+2+j],j,'test','input')
        plot_one(task, axs[1,i+2+j],j,'test','output')

    if 'solutions' in task:
        axs[0,i+j+3].axis('off'); axs[1,i+j+3].axis('off')
        for k in range(len(task['solutions'])):
            plot_one(task, axs[0,i+j+4+k],k,'solutions','input')
            plot_one(task, axs[1,i+j+4+k],k,'solutions','output')

    plt.show()


#####
##### END   src_james/plot.py
#####

#####
##### START src_james/original/Solver.py
#####

from copy import deepcopy
from itertools import chain
from typing import List, Dict

import numpy as np

# from src_james.core.DataModel import Problem
# from src_james.plot import plot_task



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
    def solve_lambda(_task_, _function_, *args, _inplace_=False, **kwargs) -> List[Dict[str,Problem]]:
        solutions = []
        for index, problem in enumerate(_task_['test']):
            output = _function_(problem['input'], *args, **kwargs)
            output.flags.writeable = False
            solution = Problem({
                "input":  problem['input'],
                "output": output,
            }, problemset=_task_['test'])
            solutions.append(problem)
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



#####
##### END   src_james/original/Solver.py
#####

#####
##### START src_james/original/GeometrySolver.py
#####

from itertools import combinations, product

import numpy as np

# from src_james.original.Solver import Solver



class GeometrySolver(Solver):
    optimise = True
    verbose  = True
    actions = {
        "flip":      ( np.flip,      [0,1]    ),
        "rot90":     ( np.rot90,     [1,2,3]  ),
        "roll":      ( np.roll,      product(range(-5,5),[0,1]) ),
        "swapaxes":  ( np.swapaxes,  [(0, 1),(1, 0)] ),
        "transpose": ( np.transpose, []       ),                      # this doesn't find anything
        }

    def __init__(self):
        super().__init__()
        for key, (function, arglist) in self.actions.items():
            self.actions[key] = (function, [ (args,) if not isinstance(args, tuple) else args for args in arglist ])

    def detect(self, task):
        return self.is_task_shape_ratio_unchanged(task)  # grids must remain the exact same size

    def test(self, task):
        if task.filename in self.cache: return True

        max_roll = (self.task_grid_max_dim(task) + 1) // 2
        for key, (function, arglist) in self.actions.items():
            if function == np.roll: arglist = product(range(-max_roll,max_roll),[0,1])
            for args in arglist:
                if self.is_lambda_valid(task, function, *args):
                    self.cache[task.filename] = (function, args)
                    if self.verbose: print(function, args)
                    return True

        # this doesn't find anything
        if self.optimise: return False
        for ((function1, arglist1),(function2, arglist2)) in combinations( self.actions.values(), 2 ):
            if function1 == np.roll: arglist1 = product(range(-max_roll,max_roll),[0,1])
            if function2 == np.roll: arglist2 = product(range(-max_roll,max_roll),[0,1])
            for args1, args2 in product(arglist1, arglist2):
                function = lambda grid, args1, args2: function1(function2(grid, *args2), *args1)
                if self.is_lambda_valid(task, function, *(args1,args2)):
                    self.cache[task.filename] = (function, (args1,args2))
                    if self.verbose: print(function, (args1,args2))
                    return True
        return False

    def action(self, grid, function=None, args=None, task=None):
        try:
            return function(grid, *args)
        except Exception as exception:
            if self.verbose: print(function, args, exception)
            return grid


#####
##### END   src_james/original/GeometrySolver.py
#####

#####
##### START src_james/original/ZoomSolver.py
#####

import cv2
import skimage.measure

# from src_james.original.Solver import Solver



class ZoomSolver(Solver):
    verbose = False

    def detect(self, task):
        ratios = self.task_shape_ratios(task)
        ratio  = list(ratios)[0]
        detect = (
                ratios != { (1,1) }   # not no scaling
                and len(ratios) == 1      # not multiple scalings
                and ratio[0] == ratio[1]  # single consistant scaling
        )
        return detect

    def get_scale(self, task):
        return list(self.task_shape_ratios(task))[0][0]

    def action( self, grid, task=None, *args ):
        scale = self.get_scale(task)
        if scale > 1:
            resize = tuple( int(d*scale) for d in grid.shape )
            output = cv2.resize(grid, resize, interpolation=cv2.INTER_NEAREST)
        else:
            resize = tuple( int(1/scale) for d in grid.shape )
            output = skimage.measure.block_reduce(grid, resize)
        if self.verbose:
            print('scale', scale, 'grid.shape', grid.shape, 'output.shape', output.shape)
            print('grid', grid)
            print('output', output)
        return output


#####
##### END   src_james/original/ZoomSolver.py
#####

#####
##### START src_james/original/functions.py
#####

import numpy as np



# from skimage.measure import block_reduce
# from numpy_lru_cache_decorator import np_cache  # https://gist.github.com/Susensio/61f4fee01150caaac1e10fc5f005eb75

def query_true(grid,x,y):          return True
def query_not_zero(grid,x,y):      return grid[x,y]
def query_color(grid,x,y,color):   return grid[x,y] == color

# evaluation/15696249.json - max(1d.argmax())
def query_max_color(grid,x,y,exclude_zero=True):
    return grid[x,y] == max_color(grid, exclude_zero)
# @lru_cache(1024)
def max_color(grid, exclude_zero=True):
    bincount = np.bincount(grid.flatten())
    if exclude_zero:
        bincount[0] = np.min(bincount)  # exclude 0
    return bincount.argmax()

def query_min_color(grid,x,y, exclude_zero=True):
    return grid[x,y] == min_color(grid, exclude_zero)
# @lru_cache(1024)
def min_color(grid,exclude_zero=True):
    bincount = np.bincount(grid.flatten())
    if exclude_zero:
        bincount[0] = np.max(bincount)  # exclude 0
    return bincount.argmin()

def query_max_color_1d(grid,x,y,exclude_zero=True):
    return grid[x,y] == max_color_1d(grid)
# @lru_cache(16)
def max_color_1d(grid,exclude_zero=True):
    return max(
        [ max_color(row,exclude_zero) for row in grid ] +
        [ max_color(col,exclude_zero) for col in np.swapaxes(grid, 0,1) ]
        )
def query_min_color_1d(grid,x,y):
    return grid[x,y] == min_color_1d(grid)
# @lru_cache(16)
def min_color_1d(grid):
    return min(
        [ min_color(row) for row in grid ] +
        [ min_color(col) for col in np.swapaxes(grid, 0,1) ]
        )


def query_count_colors(grid,x,y):
    return grid[x,y] >= count_colors(grid)
def query_count_colors_row(grid,x,y):
    return x + len(grid.shape[0])*y <= count_colors(grid)
def query_count_colors_col(grid,x,y):
    return y + len(grid.shape[1])*x <= count_colors(grid)
# @lru_cache(16)
def count_colors(grid):
    bincount = np.bincount(grid.flatten())
    return np.count_nonzero(bincount[1:]) # exclude 0

def query_count_squares(grid,x,y):
    return grid[x,y] >= count_squares(grid)
def query_count_squares_row(grid,x,y):
    return x + len(grid.shape[0])*y <= count_squares(grid)
def query_count_squares_col(grid,x,y):
    return y + len(grid.shape[1])*x <= count_squares(grid)
# @lru_cache(16)
def count_squares(grid):
    return np.count_nonzero(grid.flatten())

# BROKEN?
def rotate_loop(grid, start=0):
    angle = start
    while True:
        yield np.rot90(grid, angle % 4)
        angle += 1 * np.sign(start)

# BROKEN?
def rotate_loop_rows(grid, start=0):
    angle = start
    while True:
        yield np.rot90(grid, angle % grid.shape[0])
        angle += 1 * np.sign(start)

# BROKEN?
def rotate_loop_cols(grid, start=0):
    angle = start
    while True:
        yield np.rot90(grid, angle % grid.shape[1])
        angle += 1 * np.sign(start)

def flip_loop(grid, start=0):
    angle = start
    while True:
        if angle % 2: yield np.flip(grid)
        else:         yield grid
        angle += 1 * np.sign(start)

# BROKEN?
def flip_loop_rows(grid, start=0):
    angle = start
    while True:
        if angle % grid.shape[0]: yield np.flip(grid)
        else:                     yield grid
        angle += 1 * np.sign(start)

# BROKEN?
def flip_loop_cols(grid, start=0):
    angle = start
    while True:
        if angle % grid.shape[1]: yield np.flip(grid)
        else:                     yield grid
        angle += 1 * np.sign(start)

# BROKEN?
def invert(grid, color=None):
    if callable(color): color = color(grid)
    if color is None:   color = max_color(grid)
    if color:
        grid        = grid.copy()
        mask_zero   = grid[ grid == 0 ]
        mask_square = grid[ grid != 0 ]
        grid[mask_zero]   = color
        grid[mask_square] = 0
    return grid



# Source: https://codereview.stackexchange.com/questions/132914/crop-black-border-of-image-using-numpy
def crop_inner(grid,tol=0):
    mask = grid > tol
    return grid[np.ix_(mask.any(1),mask.any(0))]

def crop_outer(grid,tol=0):
    mask = grid>tol
    m,n  = grid.shape
    mask0,mask1 = mask.any(0),mask.any(1)
    col_start,col_end = mask0.argmax(),n-mask0[::-1].argmax()
    row_start,row_end = mask1.argmax(),m-mask1[::-1].argmax()
    return grid[row_start:row_end,col_start:col_end]

def make_tuple(args):
    if isinstance(args, tuple): return args
    if isinstance(args, list):  return tuple(args)
    return (args,)


#####
##### END   src_james/original/functions.py
#####

#####
##### START src_james/original/BorderSolver.py
#####

import numpy as np

# from src_james.original.Solver import Solver
# from src_james.original.functions import count_colors, count_squares, max_color, max_color_1d, min_color, min_color_1d



class BorderSolver(Solver):
    verbose = True
    debug = True
    cache = {}
    queries = [
        *range(0,10),
        max_color,     # FAIL: evaluation/fc754716.json
        max_color_1d,
        min_color,
        min_color_1d,
        count_colors,
        count_squares,
        np.count_nonzero,
    ]

    def task_has_border(self, task):
        if not self.is_task_shape_ratio_consistant(task): return False
        return all([ self.grid_has_border(spec['output']) for spec in task['train'] ])

    def grid_has_border(self, grid):
        if min(grid.shape) <= 2: return False  # single color problem

        grid_center = grid[1:-1,1:-1]
        return np.count_nonzero(grid_center) == 0 and all([
            np.count_nonzero(border) == len(border)
            for border in [ grid[0,:], grid[-1,:], grid[:,0], grid[:,-1] ]
        ])

    def detect(self, task):
        return self.task_has_border(task)

    def test(self, task):
        if task.filename in self.cache: return True
        for query in self.queries:
            args = [ query ]
            if self.is_lambda_valid(task, self.action, *args, task=task):
                self.cache[task.filename] = args
                if self.verbose: print(self.action, args)
                return True
        return False

    def action(self, grid, query=None, task=None):
        ratio  = list(self.task_shape_ratios(task))[0]
        output = np.zeros(( int(grid.shape[0] * ratio[0]), int(grid.shape[1] * ratio[1]) ))
        color  = query(grid) if callable(query) else query
        output[:,:] = color
        output[1:-1,1:-1] = 0
        return output


#####
##### END   src_james/original/BorderSolver.py
#####

#####
##### START src_james/original/DoNothingSolver.py
#####

# from src_james.original.Solver import Solver



class DoNothingSolver(Solver):
    def action( self, grid, task=None, *args ):
        return grid


#####
##### END   src_james/original/DoNothingSolver.py
#####

#####
##### START src_james/original/SingleColorSolver.py
#####

import numpy as np

# from src_james.original.Solver import Solver
# from src_james.original.functions import count_colors, count_squares, max_color, max_color_1d, min_color, min_color_1d



class SingleColorSolver(Solver):
    verbose = True
    debug = True
    cache = {}
    queries = [
        *range(0,10),
        max_color,     # FAIL: evaluation/fc754716.json
        max_color_1d,
        min_color,
        min_color_1d,
        count_colors,
        count_squares,
        np.count_nonzero,
    ]

    def grid_unique_colors(self, grid):
        return np.unique(grid.flatten())

    def task_is_singlecolor(self, task):
        if not self.is_task_shape_ratio_consistant(task): return False
        return all([ len(self.grid_unique_colors(spec['output'])) == 1 for spec in task['train'] ])

    def detect(self, task):
        return self.task_is_singlecolor(task)

    def test(self, task):
        if task.filename in self.cache: return True
        for query in self.queries:
            args = [ query ]
            if self.is_lambda_valid(task, self.action, *args, task=task):
                self.cache[task.filename] = args
                if self.verbose: print(self.action, args)
                return True
        return False

    def action(self, grid, query=None, task=None):
        ratio  = list(self.task_shape_ratios(task))[0]
        output = np.zeros(( int(grid.shape[0] * ratio[0]), int(grid.shape[1] * ratio[1]) ))
        color  = query(grid) if callable(query) else query
        output[:,:] = color
        return output


#####
##### END   src_james/original/SingleColorSolver.py
#####

#####
##### START src_james/original/TessellationSolver.py
#####

import inspect
from itertools import product

# from src_james.original.GeometrySolver import GeometrySolver
# from src_james.original.ZoomSolver import ZoomSolver
# from src_james.original.functions import *
# from src_james.original.functions import crop_inner, crop_outer



class TessellationSolver(GeometrySolver):
    verbose = True
    debug   = False
    options = {
        "preprocess": {
            "np.copy":    (np.copy, []),
            "crop_inner": (crop_inner, range(0,9)),
            "crop_outer": (crop_outer, range(0,9)),
            },
        "transform": {
            "none":              ( np.copy,      []        ),
            "flip":              ( np.flip,      [0,1]     ),
            "rot90":             ( np.rot90,     [1,2,3]   ),
            "roll":              ( np.roll,      product([-1,1],[0,1]) ),
            "swapaxes":          ( np.swapaxes,  [(0, 1)]  ),
            "rotate_loop":       ( rotate_loop,         range(-4,4) ),
            "rotate_loop_rows":  ( rotate_loop_rows,    range(-4,4) ),  # BROKEN ?
            "rotate_loop_cols":  ( rotate_loop_cols,    range(-4,4) ),  # BROKEN ?
            "flip_loop":         ( flip_loop,           range(0,2)  ),  # BROKEN ?
            "flip_loop_rows":    ( flip_loop_rows,      range(0,2)  ),  # BROKEN ?
            "flip_loop_cols":    ( flip_loop_cols,      range(0,2)  ),  # BROKEN ?
            "invert":            ( invert,              [max_color, min_color, max_color_1d, count_colors, count_squares, *range(1,9)]  ), # BROKEN
            # TODO: Invert
            },
        "query": {
            "query_true":              ( query_true,          [] ),
            "query_not_zero":          ( query_not_zero,      [] ),
            "query_max_color":         ( query_max_color,     [] ),
            "query_min_color":         ( query_min_color,     [] ),
            "query_max_color_1d":      ( query_max_color_1d,  [] ),
            "query_min_color_1d":      ( query_min_color_1d,  [] ),
            "query_count_colors":      ( query_count_colors,      [] ),
            "query_count_colors_row":  ( query_count_colors_row,  [] ),
            "query_count_colors_col":  ( query_count_colors_col,  [] ),
            "query_count_squares":     ( query_count_squares,     [] ),
            "query_count_squares_row": ( query_count_squares_row, [] ),
            "query_count_squares_col": ( query_count_squares_col, [] ),
            "query_color":             ( query_color,         range(0,10) ),  # TODO: query_max_color() / query_min_color()
            }
        }


    def detect(self, task):
        if self.is_task_shape_ratio_unchanged(task):            return False  # Use GeometrySolver
        if not self.is_task_shape_ratio_integer_multiple(task): return False  # Not a Tessalation problem
        if not all([ count_colors(spec['input']) == count_colors(spec['output']) for spec in task['train'] ]): return False  # Different colors
        if ZoomSolver().solve(task):                            return False
        #if not self.is_task_shape_ratio_consistant(task):       return False  # Some inconsistant grids are tessalations
        return True



    def loop_options(self):
        for (preprocess,p_args) in self.options['preprocess'].values():
            # print( (preprocess,p_args) )
            for p_arg in p_args or [()]:
                p_arg = make_tuple(p_arg)
                # print( (preprocess,p_args) )
                for (transform,t_args) in self.options['transform'].values():
                    for t_arg in t_args or [()]:
                        t_arg = make_tuple(t_arg)
                        for (query,q_args) in self.options['query'].values():
                            for q_arg in q_args or [()]:
                                q_arg = make_tuple(q_arg)
                                yield (preprocess, p_arg),(transform,t_arg),(query,q_arg)

    # TODO: hieraracharical nesting of solves and solutions/rules array generator
    def test(self, task):
        if task.filename in self.cache: return True
        for (preprocess,p_arg),(transform,t_arg),(query,q_arg) in self.loop_options():
            kwargs = {
                "preprocess": preprocess,
                "p_arg":      p_arg,
                "transform":  transform,  # TODO: invert every other row | copy pattern from train outputs | extend lines
                "t_arg":      t_arg,
                "query":      query,  # TODO: max_colour limit counter
                "q_arg":      q_arg,
                }
            if self.is_lambda_valid(task, self.action, **kwargs, task=task):
                self.cache[task.filename] = kwargs
                if self.verbose: print(self.action, kwargs)
                return True
        return False


    def action(self, grid, preprocess=np.copy, p_arg=(), transform=np.copy, t_arg=(), query=query_true, q_arg=(), task=None):
        #print('action', preprocess, transform, query)
        if inspect.isgeneratorfunction(transform):
            generator = transform(grid, *t_arg)
            transform = lambda grid, *args: next(generator)

        # Some combinations of functions will throw gemoetry
        output = None
        try:
            grid    = preprocess(grid, *p_arg)
            output  = self.get_output_grid(grid, task).copy()
            ratio   = ( int(output.shape[0] / grid.shape[0]), int(output.shape[1] / grid.shape[1]) )
            (gx,gy) = grid.shape
            for x,y in product(range(ratio[0]),range(ratio[1])):
                copy = np.zeros(grid.shape)
                # noinspection PyArgumentList
                if query(grid,x%gx,y%gy, *q_arg):
                    copy = transform(grid, *t_arg)

                output[x*gx:(x+1)*gx, y*gy:(y+1)*gy] = copy
        except Exception as exception:
            if self.debug: print(exception)
        return output


    def loop_ratio(self, task):
        ratio = list(self.task_shape_ratios(task))[0]
        for i in range(int(ratio[0])):
            for j in range(int(ratio[1])):
                yield i,j


    def get_output_grid(self, grid, task):
        try:
            #print('get_output_grid(self, grid, task)', grid, task)
            for index, spec in enumerate(task['train']):
                if spec['input'] is grid:
                    return spec['output']
            else:
                # No output for tests
                ratio = list(self.task_shape_ratios(task))[0]
                ratio = list(map(int, ratio))
                shape = ( int(grid.shape[0]*ratio[0]), int(grid.shape[1]*ratio[1]) )
                return np.zeros(shape)
        except Exception as exception:
            if self.debug: print(exception)
            pass


#####
##### END   src_james/original/TessellationSolver.py
#####

#####
##### START src_james/original/solvers.py
#####

from typing import List

# from src_james.original.BorderSolver import BorderSolver
# from src_james.original.DoNothingSolver import DoNothingSolver
# from src_james.original.GeometrySolver import GeometrySolver
# from src_james.original.SingleColorSolver import SingleColorSolver
# from src_james.original.Solver import Solver
# from src_james.original.TessellationSolver import TessellationSolver
# from src_james.original.ZoomSolver import ZoomSolver

solvers: List[Solver] = [
    DoNothingSolver(),
    BorderSolver(),
    GeometrySolver(),
    SingleColorSolver(),
    TessellationSolver(),
    ZoomSolver(),
]


#####
##### END   src_james/original/solvers.py
#####

#####
##### START ./src_james/original/main.py
#####

import os
import time

# from src_james.core.DataModel import Competition
# from src_james.original.solvers import solvers



time_start   = time.perf_counter()
competition  = Competition()
scores       = { name: 0 for name in competition.values() }
for name, dataset in competition.items():
    time_start_dataset = time.perf_counter()
    for solver in solvers:
        solver.cache = {}
        if os.environ.get('KAGGLE_KERNEL_RUN_TYPE', ''):
            scores[name] = solver.solve_all(dataset)
        else:
            scores[name] = solver.plot(dataset)

    dataset.time_taken = time.perf_counter() - time_start_dataset
competition.time_taken = time.perf_counter() - time_start

competition['test'].write_submission()

print('-'*20)
print('Score:')
for key, value in competition.score().items(): print(f'{key.rjust(11)}: {value}')


#####
##### END   ./src_james/original/main.py
#####

##### 
##### ./submission/kaggle_compile.py ./src_james/original/main.py
##### 
##### 2020-05-22 22:41:08+01:00
##### 
##### origin	git@github.com:seshurajup/kaggle-arc.git (fetch)
##### origin	git@github.com:seshurajup/kaggle-arc.git (push)
##### 
#####   james-wip c81cf89 Solvers | work in progress - broken
##### * master    08f4dab import original notebook codebase
##### 
##### 08f4dabd9a073fb1d962168fed1d5751eae5572a
##### 
