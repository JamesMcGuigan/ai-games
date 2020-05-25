#!/usr/bin/env python3

##### 
##### ./submission/kaggle_compile.py ./src_james/solver_multimodel/main.py
##### 
##### 2020-05-26 00:22:14+01:00
##### 
##### origin	git@github.com:seshurajup/kaggle-arc.git (fetch)
##### origin	git@github.com:seshurajup/kaggle-arc.git (push)
##### 
##### * master ccab583 [ahead 1] GlobSover | Create a lookup table of all previously seen input/output pairs
##### 
##### ccab58302b388ed65c55c9e12812429893ec29cb
##### 

#####
##### START src_james/settings.py
#####

# DOCS: https://www.kaggle.com/WinningModelDocumentationGuidelines
import os
import pathlib
try:    root_dir = pathlib.Path(__file__).parent.parent.absolute()
except: root_dir = ''

settings = {
    'production': os.environ.get('KAGGLE_KERNEL_RUN_TYPE', None) != None or 'submission' in __file__
}
settings = {
    **settings,
    'verbose': True,
    'debug':   not settings['production'],
    'caching': settings['production'] or True,
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

    ### No need to extend sample_submission.csv, just sort the CSV
    # @classmethod
    # def sample_submission(cls):
    #     filename = os.path.join(settings['dir']['data'],'sample_submission.csv')
    #     sample_submission = pd.read_csv(filename)
    #     return sample_submission
    #
    # @classmethod
    # def write_submission(cls, dataset: 'Dataset', filename='submission.csv'):
    #     csv        = CSV.to_csv(dataset)
    #     lines      = csv.split('\n')
    #     line_count = len(lines)
    #     data       = []
    #
    #     submission = cls.sample_submission()
    #     submission = submission.set_index('output_id', drop=False)
    #     for line in lines[1:]:  # ignore header
    #         object_id,output = line.split(',',2)
    #         submission.loc[object_id]['output'] = output
    #
    #     submission.to_csv(filename, index=False)
    #     print(f"\nwrote: {filename} | {line_count} lines")


    @classmethod
    def object_id(cls, filename, index=0) -> str:
        return re.sub('^.*/|\.json$', '', filename) + '_' + str(index)

    @classmethod
    def to_csv(cls, dataset: 'Dataset'):
        csv = []
        for task in dataset:
            line = CSV.to_csv_line(task)
            if line: csv.append(line)
        csv = ['output_id,output'] + sorted(csv) # object_id keys are sorted in sample_submission.csv
        return "\n".join(csv)

    @classmethod
    def default_csv_line(cls, task: 'Task') -> str:
        return '|123|456|789|'

    @classmethod
    def to_csv_line(cls, task: 'Task') -> str:
        csv = []
        for index, problemset in enumerate(task['solutions']):
            solutions = list(set(
                cls.grid_to_csv_string(problem['output'])
                for problem in problemset
            ))
            solution_str = " ".join(solutions[:3]) if len(solutions) else cls.default_csv_line(task)
            line = ",".join([
                cls.object_id(task.filename, index),
                solution_str
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
        value = np.ascontiguousarray(value, dtype=self.dtype)
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



#####
##### END   src_james/core/DataModel.py
#####

#####
##### START src_james/util/np_cache.py
#####

# Inspired by: https://stackoverflow.com/questions/52331944/cache-decorator-for-numpy-arrays/52332109
from functools import wraps

import numpy as np
from fastcache._lrucache import clru_cache

### Profiler: 2x speedup
# from src_james.settings import settings

__np_cache = {}
def np_cache(maxsize=1024, typed=True):
    """
        Decorator:
        @np_cache
        def fn(): return value

        @np_cache(maxsize=128, typed=True)
        def fn(): return value
    """
    maxsize_default=None

    def np_cache_generator(function):
        if not settings['caching']: return function
        @wraps(function)
        def wrapper(*args, **kwargs):
            ### def encode(*args, **kwargs):
            args = list(args)  # BUGFIX: TypeError: 'tuple' object does not support item assignment
            for i, arg in enumerate(args):
                if isinstance(arg, np.ndarray):
                    hash = arg.tobytes()
                    if hash not in wrapper.cache:
                        wrapper.cache[hash] = arg
                    args[i] = hash
            for key, arg in kwargs.items():
                if isinstance(arg, np.ndarray):
                    hash = arg.tobytes()
                    if hash not in wrapper.cache:
                        wrapper.cache[hash] = arg
                    kwargs[key] = hash

            return cached_wrapper(*args, **kwargs)

        @clru_cache(maxsize=maxsize, typed=typed)
        def cached_wrapper(*args, **kwargs):
            ### def decode(*args, **kwargs):
            args = list(args)  # BUGFIX: TypeError: 'tuple' object does not support item assignment
            for i, arg in enumerate(args):
                if isinstance(arg, bytes) and arg in wrapper.cache:
                    args[i] = wrapper.cache[arg]
            for key, arg in kwargs.items():
                if isinstance(arg, bytes) and arg in wrapper.cache:
                    kwargs[key] = wrapper.cache[arg]

            return function(*args, **kwargs)

        # copy lru_cache attributes over too
        wrapper.cache       = __np_cache  # use a shared cache between wrappers to save memory
        wrapper.cache_info  = cached_wrapper.cache_info
        wrapper.cache_clear = cached_wrapper.cache_clear

        return wrapper


    ### def np_cache(maxsize=1024, typed=True):
    if callable(maxsize):
        (function, maxsize) = (maxsize, maxsize_default)
        return np_cache_generator(function)
    else:
        return np_cache_generator

#####
##### END   src_james/util/np_cache.py
#####

#####
##### START src_james/plot.py
#####

# Source: https://www.kaggle.com/jamesmcguigan/arc-geometry-solvers/
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from fastcache._lrucache import clru_cache
from matplotlib import colors

# Modified from: https://www.kaggle.com/zaharch/visualizing-all-tasks-updated
# from src_james.core.DataModel import Task


@clru_cache()
def invert_hexcode(hexcode):
    hexcode = hexcode.replace('#','0x')
    number  = (16**len(hexcode)-1) - int(hexcode, 16)
    return hex(number).replace('0x','#')

def plot_one(task, ax, i,train_or_test,input_or_output):
    hexcodes = [
        '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25',
    ]
    # inverted_hexcodes = list(map(invert_hexcode,hexcodes))
    # icmap = colors.ListedColormap(inverted_hexcodes)
    cmap  = colors.ListedColormap(hexcodes)
    norm  = colors.Normalize(vmin=0, vmax=9)

    try:
        input_matrix  = task[train_or_test][i][input_or_output]
        font_size     = 50 / np.sqrt(input_matrix.shape[0] * input_matrix.shape[1])
        min_font_size = 6

        ax.imshow(input_matrix, cmap=cmap, norm=norm)
        # DOC: https://stackoverflow.com/questions/33828780/matplotlib-display-array-values-with-imshow
        if font_size >= min_font_size:
            for (j,i),label in np.ndenumerate(input_matrix):
                ax.text(i,j,label,ha='center',va='center', fontsize=font_size, color='black')
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
    task_solutions = {
        "solutions": list(chain(*task['solutions']))  # this is a 2D array now
    }
    num_train      = len(task['train']) + len(task['test']) + 1
    if task.solutions_count: num_train += task.solutions_count + 1

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

    if task.solutions_count:
        axs[0,i+j+3].axis('off'); axs[1,i+j+3].axis('off')
        for k in range(len(task_solutions)):
            plot_one(task_solutions, axs[0,i+j+4+k],k,'solutions','input')
            plot_one(task_solutions, axs[1,i+j+4+k],k,'solutions','output')

    for ax in chain(*axs): ax.axis('off')
    plt.show()


#####
##### END   src_james/plot.py
#####

#####
##### START src_james/solver_multimodel/Solver.py
#####

from typing import List, Union, Callable

import numpy as np

# from src_james.core.DataModel import Problem, Task, Dataset
# from src_james.plot import plot_task


class Solver():
    verbose = False
    debug   = False
    def __init__(self):
        self.cache = {}

    @staticmethod
    def is_lambda_valid(_task_: Task, _function_: Callable, *args, **kwargs):  # _task_ = avoid namespace conflicts with kwargs=task
        for problem in _task_['train']:
            output = _function_(problem['input'], *args, **kwargs)
            if not np.array_equal( problem['output'], output):
                return False
        return True

    @staticmethod
    def solve_lambda( _task_: Task, _function_: Callable, *args, _inplace_=False, **kwargs) -> List[Problem]:
        solutions = []
        for index, problem in enumerate(_task_['test']):
            output = _function_(problem['input'], *args, **kwargs)
            output.flags.writeable = False
            solution = Problem({
                "input":  problem['input'],
                "output": output,
            }, problemset=_task_['test'])
            solutions.append(solution)
            if _inplace_:
                _task_['solutions'][index].append(solution)
        return solutions

    def action(self, grid: np.ndarray, task=None, *args):
        """This is the primary method this needs to be defined"""
        return grid
        # raise NotImplementedError()

    def detect(self, task: Task) -> bool:
        """default heuristic is simply to run the solver"""
        return self.test(task)

    def fit(self, task: Task):
        if task.filename in self.cache: return
        pass

    def test(self, task: Task) -> bool:
        """test if the given action correctly solves the task"""
        if task.filename not in self.cache: self.fit(task)

        args = self.cache.get(task.filename, ())
        return self.is_lambda_valid(task, self.action, *args, task=task)


    def format_args(self, args):
        if isinstance(args, dict):
            args = dict(zip(args.keys(), map(self.format_args, list(args.values()))))
        elif isinstance(args, (list,set,tuple)):
            args = list(args)
            for index, arg in enumerate(args):
                if hasattr(arg, '__name__'):
                    arg = f"<{type(arg).__name__}:{arg.__name__}>"
                if isinstance(arg, (list,set,tuple,dict)):
                    arg = self.format_args(arg)
                args[index] = arg
            args = tuple(args)
        return args

    def log_solved(self, task: Task, args: Union[list,tuple,set], solutions: List[Problem]):
        if self.verbose:
            if 'test' in task.filename:           label = 'test  '
            elif self.is_solved(task, solutions): label = 'solved'
            else:                                 label = 'guess '

            args  = self.format_args(args) if len(args) else None
            print(f'{label}:', task.filename, self.__class__.__name__, args)

    def is_solved(self, task: Task, solutions: List[Problem]):
        for solution in solutions:
            for problem in task['test']:
                if solution == problem:
                    return True
        return False

    def solve(self, task: Task, force=False, inplace=True) -> Union[List[Problem],None]:
        """solve test case and persist"""
        if task.filename not in self.cache: self.fit(task)
        try:
            if self.detect(task) or force:    # may generate cache
                if self.test(task) or force:  # may generate cache
                    args = self.cache.get(task.filename, ())
                    if isinstance(args, dict):
                        solutions = self.solve_lambda(task, self.action, **args, task=task, _inplace_=True)
                    else:
                        solutions = self.solve_lambda(task, self.action,  *args, task=task, _inplace_=True )
                    if len(solutions):
                        self.log_solved(task, args, solutions)
                    return solutions
        except Exception as exception:
            if self.debug: raise exception
        return None

    def solve_all(self, tasks: Union[Dataset,List[Task]], plot=False, solve_detects=False):
        count = 0
        for task in tasks:
            if self.detect(task):
                solution = self.solve(task, force=solve_detects)
                if solution or (solve_detects and self.test(task)):
                    count += 1
                    if plot == True:
                        plot_task(task)
        return count

    def plot(self, tasks: Union[Dataset,List[Task], Task]):
        if isinstance(tasks, Task): tasks = [ tasks ]
        return self.solve_all(tasks, plot=True, solve_detects=False)

    def plot_detects(self, tasks: Union[Dataset,List[Task],Task], unsolved=True):
        if isinstance(tasks, Task): tasks = [ tasks ]
        if unsolved:
            tasks = [ task for task in tasks if not task.solutions_count ]
        return self.solve_all(tasks, plot=True, solve_detects=True)


#####
##### END   src_james/solver_multimodel/Solver.py
#####

#####
##### START src_james/solver_multimodel/queries/ratio.py
#####

from itertools import chain
from typing import List

import numpy as np
from fastcache._lrucache import clru_cache

# from src_james.core.DataModel import Task
# from src_james.util.np_cache import np_cache


@np_cache
def grid_shape_ratio(grid1, grid2):
    try:
        return ( grid2.shape[0] / grid1.shape[0], grid2.shape[1] / grid1.shape[1] )
    except:
        return (0, 0)  # For tests

@clru_cache()
def task_grids(task):
    grids = []
    for test_train in ['test','train']:
        for spec in task[test_train]:
            grids += [ spec.get('input',[]), spec.get('output',[]) ]  # tests not gaurenteed to have outputs
    return grids

@clru_cache()
def task_grid_shapes(task):
    return [ np.array(grid).shape for grid in task_grids(task) ]

@clru_cache()
def task_grid_max_dim(task):
    return max(chain(*task_grid_shapes(task)))

@clru_cache()
def is_task_shape_ratio_unchanged(task):
    return task_shape_ratios(task) == [ (1,1) ]

@clru_cache()
def is_task_shape_ratio_consistant(task):
    return len(task_shape_ratios(task)) == 1

@clru_cache()
def is_task_shape_ratio_integer_multiple(task):
    ratios = task_shape_ratios(task)
    return all([ isinstance(d, int) or d.is_integer() for d in chain(*ratios) ])

@clru_cache()
def task_shape_ratios(task: Task) -> List:
    ratios = list(set([
        grid_shape_ratio(problem.get('input',[]), problem.get('output',[]))
        for problem in task['train']
    ]))
    # ratios = set([ int(ratio) if ratio.is_integer() else ratio for ratio in chain(*ratios) ])
    return ratios


@clru_cache()
def is_task_shape_ratio_consistent(task):
    return len(task_shape_ratios(task)) == 1

@clru_cache()
def is_task_shape_ratio_integer_multiple(task):
    ratios = task_shape_ratios(task)
    return all([ isinstance(d, int) or d.is_integer() for d in chain(*ratios) ])



#####
##### END   src_james/solver_multimodel/queries/ratio.py
#####

#####
##### START src_james/ensemble/util.py
#####

import numpy as np


def Defensive_Copy(A):
    n = len(A)
    k = len(A[0])
    L = np.zeros((n, k), dtype=np.int8)
    for i in range(n):
        for j in range(k):
            L[i, j] = 0 + A[i][j]
    return L.tolist()


def Create(task, task_id=0):
    n = len(task['train'])
    Input  = [Defensive_Copy(task['train'][i]['input'])  for i in range(n)]
    Output = [Defensive_Copy(task['train'][i]['output']) for i in range(n)]
    Input.append(Defensive_Copy(task['test'][task_id]['input']))
    return Input, Output


def flattener(pred):
    if pred is None: return ''
    pred = np.array(pred).astype(np.int8).tolist()
    str_pred = str([row for row in pred])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred


#####
##### END   src_james/ensemble/util.py
#####

#####
##### START src_james/ensemble/period.py
#####

import numpy as np

# from src_james.ensemble.util import Defensive_Copy
# from src_james.util.np_cache import np_cache


@np_cache()
def get_period_length0(arr):
    H, W = arr.shape
    period = 1
    while True:
        cycled = np.pad(arr[:period, :], ((0, H - period), (0, 0)), 'wrap')
        if (cycled == arr).all():
            return period
        period += 1

@np_cache()
def get_period_length1(arr):
    H, W = arr.shape
    period = 1
    while True:
        cycled = np.pad(arr[:, :period], ((0, 0), (0, W - period)), 'wrap')
        if (cycled == arr).all():
            return period
        period += 1


def get_period(arr0):
    if np.sum(arr0) == 0:
        return -1
    #     arr_crop=get_bound_image(arr0)
    #     arr=np.array(arr_crop)
    arr = np.array(arr0)
    a, b = get_period_length0(arr), get_period_length1(arr)
    period = arr[:a, :b]
    if period.shape == arr.shape:
        return -1
    return period.tolist()


def same_ratio(basic_task):
    # returns -1 if no match is found
    # returns Transformed_Test_Case  if the mathching rule is found
    # for this notebook we only look at mosaics
    Input  = [ Defensive_Copy(x) for x in basic_task[0] ]
    Output = [ Defensive_Copy(y) for y in basic_task[1] ]
    same_ratio = True
    R_x = []
    R_y = []
    for x, y in zip(Input[:-1], Output):

        if x == []:
            same_ratio = False
            break

        n1 = len(x)
        n2 = len(y)
        k1 = len(x[0])
        k2 = len(y[0])

        R_y.append(n2 / n1)
        R_x.append(k2 / k1)

    if same_ratio and min(R_x) == max(R_x) and min(R_y) == max(R_y):
        r1 = min(R_y)
        r2 = min(R_x)
        return r1, r2

    return -1


#####
##### END   src_james/ensemble/period.py
#####

#####
##### START src_james/solver_multimodel/queries/grid.py
#####

import numpy as np

# from skimage.measure import block_reduce
# from numpy_lru_cache_decorator import np_cache  # https://gist.github.com/Susensio/61f4fee01150caaac1e10fc5f005eb75
# from src_james.util.np_cache import np_cache


def query_true(grid,x,y):          return True
def query_not_zero(grid,x,y):      return grid[x,y]
def query_color(grid,x,y,color):   return grid[x,y] == color


# evaluation/15696249.json - max(1d.argmax())
@np_cache
def query_max_color(grid,x,y,exclude_zero=True):
    return grid[x,y] == max_color(grid, exclude_zero)

@np_cache
def max_color(grid, exclude_zero=True):
    bincount = np.bincount(grid.flatten())
    if exclude_zero:
        bincount[0] = np.min(bincount)  # exclude 0
    return bincount.argmax()

@np_cache
def query_min_color(grid,x,y, exclude_zero=True):
    return grid[x,y] == min_color(grid, exclude_zero)

@np_cache
def min_color(grid,exclude_zero=True):
    bincount = np.bincount(grid.flatten())
    if exclude_zero:
        bincount[0] = np.max(bincount)  # exclude 0
    return bincount.argmin()

@np_cache
def query_max_color_1d(grid,x,y,exclude_zero=True):
    return grid[x,y] == max_color_1d(grid)

@np_cache
def max_color_1d(grid,exclude_zero=True):
    return max(
        [ max_color(row,exclude_zero) for row in grid ] +
        [ max_color(col,exclude_zero) for col in np.swapaxes(grid, 0,1) ]
    )

@np_cache
def query_min_color_1d(grid,x,y):
    return grid[x,y] == min_color_1d(grid)

@np_cache
def min_color_1d(grid):
    return min(
        [ min_color(row) for row in grid ] +
        [ min_color(col) for col in np.swapaxes(grid, 0,1) ]
    )

@np_cache
def query_count_colors(grid,x,y):
    return grid[x,y] >= count_colors(grid)

@np_cache
def query_count_colors_row(grid,x,y):
    return x + grid.shape[0]*y <= count_colors(grid)

@np_cache
def query_count_colors_col(grid,x,y):
    return y + grid.shape[1]*x <= count_colors(grid)


@np_cache
def count_colors(grid):
    bincount = np.bincount(grid.flatten())
    return np.count_nonzero(bincount[1:]) # exclude 0

@np_cache
def query_count_squares(grid,x,y):
    return grid[x,y] >= count_squares(grid)

@np_cache
def query_count_squares_row(grid,x,y):
    return x + grid.shape[0]*y <= count_squares(grid)

@np_cache
def query_count_squares_col(grid,x,y):
    return y + grid.shape[1]*x <= count_squares(grid)

@np_cache
def count_squares(grid):
    return np.count_nonzero(grid.flatten())

@np_cache
def grid_unique_colors(grid):
    return np.unique(grid.flatten())


#####
##### END   src_james/solver_multimodel/queries/grid.py
#####

#####
##### START src_james/solver_multimodel/GeometrySolver.py
#####

from itertools import combinations, product

import numpy as np

# from src_james.solver_multimodel.Solver import Solver
# from src_james.solver_multimodel.queries.ratio import task_grid_max_dim, is_task_shape_ratio_unchanged


class GeometrySolver(Solver):
    optimise = True
    verbose  = True
    debug    = False
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
        return is_task_shape_ratio_unchanged(task)  # grids must remain the exact same size

    def test(self, task):
        if task.filename in self.cache: return True

        max_roll = (task_grid_max_dim(task) + 1) // 2
        for key, (function, arglist) in self.actions.items():
            if function == np.roll: arglist = product(range(-max_roll,max_roll),[0,1])
            for args in arglist:
                if self.is_lambda_valid(task, function, *args):
                    self.cache[task.filename] = (function, args)
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
                    return True
        return False

    def action(self, grid, function=None, args=None, task=None):
        try:
            return function(grid, *args)
        except Exception as exception:
            if self.debug: print('Exception', self.__class__.__name__, 'action()', function, args, exception)
            return grid


#####
##### END   src_james/solver_multimodel/GeometrySolver.py
#####

#####
##### START src_james/solver_multimodel/ZoomSolver.py
#####

import cv2
import skimage.measure

# from src_james.solver_multimodel.Solver import Solver
# from src_james.solver_multimodel.queries.ratio import task_shape_ratios


class ZoomSolver(Solver):
    verbose = False

    def detect(self, task):
        ratios = task_shape_ratios(task)
        ratio  = list(ratios)[0]
        detect = (
                ratios != { (1,1) }   # not no scaling
                and len(ratios) == 1      # not multiple scalings
                and ratio[0] == ratio[1]  # single consistant scaling
        )
        return detect

    def get_scale(self, task):
        return task_shape_ratios(task)[0][0]

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
##### END   src_james/solver_multimodel/ZoomSolver.py
#####

#####
##### START src_james/solver_multimodel/queries/loops.py
#####

import numpy as np

# from src_james.solver_multimodel.queries.ratio import task_shape_ratios
# from src_james.util.np_cache import np_cache


@np_cache
def loop_ratio(task):
    ratio = list(task_shape_ratios(task))[0]
    for i in range(int(ratio[0])):
        for j in range(int(ratio[1])):
            yield i,j

# BROKEN?
@np_cache
def rotate_loop(grid, start=0):
    angle = start
    while True:
        yield np.rot90(grid, angle % 4)
        angle += 1 * np.sign(start)

# BROKEN?
@np_cache
def rotate_loop_rows(grid, start=0):
    angle = start
    while True:
        yield np.rot90(grid, angle % grid.shape[0])
        angle += 1 * np.sign(start)

# BROKEN?
@np_cache
def rotate_loop_cols(grid, start=0):
    angle = start
    while True:
        yield np.rot90(grid, angle % grid.shape[1])
        angle += 1 * np.sign(start)

@np_cache
def flip_loop(grid, start=0):
    angle = start
    while True:
        if angle % 2: yield np.flip(grid)
        else:         yield grid
        angle += 1 * np.sign(start)

# BROKEN?
@np_cache
def flip_loop_rows(grid, start=0):
    angle = start
    while True:
        if angle % grid.shape[0]: yield np.flip(grid)
        else:                     yield grid
        angle += 1 * np.sign(start)

# BROKEN?
@np_cache
def flip_loop_cols(grid, start=0):
    angle = start
    while True:
        if angle % grid.shape[1]: yield np.flip(grid)
        else:                     yield grid
        angle += 1 * np.sign(start)



#####
##### END   src_james/solver_multimodel/queries/loops.py
#####

#####
##### START src_james/solver_multimodel/transforms/crop.py
#####


# Source: https://codereview.stackexchange.com/questions/132914/crop-black-border-of-image-using-numpy
import numpy as np

# from src_james.util.np_cache import np_cache


@np_cache
def crop_inner(grid,tol=0):
    mask = grid > tol
    return grid[np.ix_(mask.any(1),mask.any(0))]

@np_cache
def crop_outer(grid,tol=0):
    mask = grid>tol
    m,n  = grid.shape
    mask0,mask1 = mask.any(0),mask.any(1)
    col_start,col_end = mask0.argmax(),n-mask0[::-1].argmax()
    row_start,row_end = mask1.argmax(),m-mask1[::-1].argmax()
    return grid[row_start:row_end,col_start:col_end]



#####
##### END   src_james/solver_multimodel/transforms/crop.py
#####

#####
##### START src_james/solver_multimodel/transforms/grid.py
#####


# BROKEN?
# from src_james.solver_multimodel.queries.grid import max_color
# from src_james.util.np_cache import np_cache


@np_cache
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


#####
##### END   src_james/solver_multimodel/transforms/grid.py
#####

#####
##### START src_james/util/make_tuple.py
#####


def make_tuple(args):
    if isinstance(args, tuple): return args
    if isinstance(args, list):  return tuple(args)
    return (args,)

#####
##### END   src_james/util/make_tuple.py
#####

#####
##### START src_james/solver_multimodel/queries/colors.py
#####

# from src_james.solver_multimodel.queries.grid import grid_unique_colors
# from src_james.solver_multimodel.queries.ratio import is_task_shape_ratio_consistant


def task_is_singlecolor(task):
    if not is_task_shape_ratio_consistant(task): return False
    return all([ len(grid_unique_colors(spec['output'])) == 1 for spec in task['train'] ])



#####
##### END   src_james/solver_multimodel/queries/colors.py
#####

#####
##### START submission/submission.py
#####



#####
##### END   submission/submission.py
#####

#####
##### START src_james/solver_multimodel/BorderSolver.py
#####

# from src_james.solver_multimodel.Solver import Solver
# from src_james.solver_multimodel.queries.grid import *
# from src_james.solver_multimodel.queries.ratio import is_task_shape_ratio_consistant, task_shape_ratios


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
        if not is_task_shape_ratio_consistant(task): return False
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
                return True
        return False

    def action(self, grid, query=None, task=None):
        ratio  = task_shape_ratios(task)[0]
        output = np.zeros(( int(grid.shape[0] * ratio[0]), int(grid.shape[1] * ratio[1]) ))
        color  = query(grid) if callable(query) else query
        output[:,:] = color
        output[1:-1,1:-1] = 0
        return output


#####
##### END   src_james/solver_multimodel/BorderSolver.py
#####

#####
##### START src_james/solver_multimodel/DoNothingSolver.py
#####

# from src_james.solver_multimodel.Solver import Solver



class DoNothingSolver(Solver):
    def action( self, grid, task=None, *args ):
        return grid


#####
##### END   src_james/solver_multimodel/DoNothingSolver.py
#####

#####
##### START src_james/solver_multimodel/GlobSolver.py
#####

# from src_james.settings import settings
# from src_james.solver_multimodel.queries.grid import *
# from src_james.solver_multimodel.Solver import Solver
# from submission.submission import Competition


class GlobSolver(Solver):
    """ Create a lookup table of all previously seen input/output pairs """
    verbose = True
    debug = True
    solutions = {}
    cache     = {}

    def __init__(self, tests_only=True):
        super().__init__()
        self.tests_only = tests_only
        self.init_cache()

    def init_cache(self):
        if len(self.cache): return
        competition = Competition()
        for name, dataset in competition.items():
            if name == 'test': continue  # exclude test from the cache
            for task in dataset:
                for name, problemset in task.items():
                    for problem in problemset:
                        try:
                            if len(problem) == 0: continue
                            if problem['input'] is None or problem['output'] is None: continue
                            hash = problem['input'].tobytes()
                            self.solutions[hash] = (task.filename, problem['output'])
                        except Exception as exception:
                            pass


    def detect(self, task):
        if task.filename in self.cache: return True
        if self.tests_only and 'test' not in task.filename: return False  # We would get 100% success rate otherwise

        # Loop through the all the inputs, as see if it is in our public database
        for name, problemset in task.items():
            inputs = [ problem['input'] for problem in problemset if problem ]
            for input in inputs:
                hash = input.tobytes()
                if hash in self.solutions:
                    filename, solutions = self.solutions[hash]
                    self.cache[task.filename] = (filename,)  # for logging purposes
                    return True
        return False


    def action(self, grid: np.ndarray, filename:str=None, task=None, *args):
        """If we have seen the input before, then propose the same output"""
        hash = grid.tobytes()
        if hash in self.solutions:
            filename, solutions = self.solutions[hash]
            return solutions
        else:
            return None


if __name__ == '__main__' and not settings['production']:
    solver = GlobSolver(tests_only=True)
    solver.verbose = True

    competition = Competition()
    competition.map(solver.solve_all)
    print(competition)

#####
##### END   src_james/solver_multimodel/GlobSolver.py
#####

#####
##### START src_james/solver_multimodel/SingleColorSolver.py
#####

import os

# from src_james.core.DataModel import Task
# from src_james.settings import settings
# from src_james.solver_multimodel.Solver import Solver
# from src_james.solver_multimodel.queries.colors import task_is_singlecolor
# from src_james.solver_multimodel.queries.grid import *
# from src_james.solver_multimodel.queries.ratio import task_shape_ratios


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

    def detect(self, task):
        return task_is_singlecolor(task)

    def fit(self, task: Task):
        if task.filename in self.cache: return True
        for query in self.queries:
            args = ( query, )
            if self.is_lambda_valid(task, self.action, *args, task=task):
                self.cache[task.filename] = args
                break

    def action(self, grid, query=None, task=None):
        ratio  = task_shape_ratios(task)[0]
        output = np.zeros(( int(grid.shape[0] * ratio[0]), int(grid.shape[1] * ratio[1]) ))
        color  = query(grid) if callable(query) else query
        output[:,:] = color
        return output




if __name__ == '__main__' and not settings['production']:
    solver = SingleColorSolver()
    solver.verbose = True
    filenames = [
        'training/5582e5ca.json',  # solved
        'training/445eab21.json',  # solved
        'training/27a28665.json',
        'training/44f52bb0.json',
        'evaluation/3194b014.json',
        'test/3194b014.json',
    ]
    for filename in filenames:
        task = Task(filename)
        solver.plot_detects([task])

    # competition = Competition()
    # # competition['test'].apply(solver.solve_all)
    # competition.map(solver.solve_all)
    # print(competition)

#####
##### END   src_james/solver_multimodel/SingleColorSolver.py
#####

#####
##### START src_james/solver_multimodel/TessellationSolver.py
#####

import inspect
import os
from itertools import product

# from src_james.core.DataModel import Task
# from src_james.settings import settings
# from src_james.solver_multimodel.GeometrySolver import GeometrySolver
# from src_james.solver_multimodel.ZoomSolver import ZoomSolver
# from src_james.solver_multimodel.queries.grid import *
# from src_james.solver_multimodel.queries.loops import *
# from src_james.solver_multimodel.queries.ratio import is_task_shape_ratio_integer_multiple
# from src_james.solver_multimodel.queries.ratio import is_task_shape_ratio_unchanged
# from src_james.solver_multimodel.transforms.crop import crop_inner, crop_outer
# from src_james.solver_multimodel.transforms.grid import invert
# from src_james.util.make_tuple import make_tuple


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
        if is_task_shape_ratio_unchanged(task):            return False  # Use GeometrySolver
        if not is_task_shape_ratio_integer_multiple(task): return False  # Not a Tessalation problem
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
                return True
        return False


    def action(self, grid, preprocess=np.copy, p_arg=(), transform=np.copy, t_arg=(), query=query_true, q_arg=(), task=None):
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


    def get_output_grid(self, grid, task):
        try:
            #print('get_output_grid(self, grid, task)', grid, task)
            for index, spec in enumerate(task['train']):
                if spec['input'] is grid:
                    return spec['output']
            else:
                # No output for tests
                ratio = task_shape_ratios(task)[0]
                ratio = list(map(int, ratio))
                shape = ( int(grid.shape[0]*ratio[0]), int(grid.shape[1]*ratio[1]) )
                return np.zeros(shape)
        except Exception as exception:
            if self.debug: print(exception)
            pass


if __name__ == '__main__' and not settings['production']:
    # This is a known test success
    task   = Task('test/27f8ce4f.json')
    solver = TessellationSolver()
    solver.plot([ task ])
    print('task.score(): ', task.score())


#####
##### END   src_james/solver_multimodel/TessellationSolver.py
#####

#####
##### START src_james/solver_multimodel/XGBSolver.py
#####

from itertools import product
from typing import List

import pydash
from fastcache._lrucache import clru_cache
from xgboost import XGBClassifier

# from src_james.core.DataModel import Competition
# from src_james.core.DataModel import Task
# from src_james.ensemble.period import get_period_length0
# from src_james.ensemble.period import get_period_length1
# from src_james.settings import settings
# from src_james.solver_multimodel.queries.grid import *
# from src_james.solver_multimodel.queries.ratio import is_task_shape_ratio_unchanged
# from src_james.solver_multimodel.Solver import Solver
# from src_james.util.np_cache import np_cache


class XGBSolver(Solver):
    optimise = True
    verbose  = True

    def __init__(self, n_estimators=24, max_depth=10, **kwargs):
        super().__init__()
        self.kwargs = { "n_estimators": n_estimators, "max_depth": max_depth, **kwargs }
        if self.kwargs.get('booster') == 'gblinear':
            self.kwargs = pydash.omit(self.kwargs, *['max_depth'])

    def __repr__(self):
        return f'<{self.__class__.__name__}:self.kwargs>'

    def format_args(self, args):
        return self.kwargs

    def detect(self, task):
        if not is_task_shape_ratio_unchanged(task): return False
        # inputs, outputs, not_valid = self.features(task)
        # if not_valid: return False
        return True

    def fit(self, task):
        if task.filename not in self.cache:
            # inputs  = task['train'].inputs + task['test'].inputs
            # outputs = task['train'].outputs
            inputs, outputs, not_valid = self.features(task)
            if not_valid:
                self.cache[task.filename] = None
            else:
                xgb = XGBClassifier(**self.kwargs, n_jobs=-1)
                xgb.fit(inputs, outputs, verbose=False)
                self.cache[task.filename] = (xgb,)

    def test(self, task: Task) -> bool:
        """test if the given action correctly solves the task"""
        args = self.cache.get(task.filename, ())
        return self.is_lambda_valid(task, self.action, *args, task=task)

    def action(self, grid, xgb=None, task=None):
        if task and task.filename not in self.cache: self.fit(task)
        xgb      = xgb or self.cache[task.filename][0]
        features = self.make_features(grid, )
        predict  = xgb.predict(features)
        output   = predict.reshape(*grid.shape)
        return output


    @classmethod
    @clru_cache(None)
    def features(cls, task, mode='train'):
        num_train_pairs = len(task[mode])
        feat, target = [], []

        for task_num in range(num_train_pairs):
            input_color              = np.array(task[mode][task_num]['input'])
            target_color             = task[mode][task_num]['output']
            nrows, ncols             = len(task[mode][task_num]['input']),  len(task[mode][task_num]['input'][0])
            target_rows, target_cols = len(task[mode][task_num]['output']), len(task[mode][task_num]['output'][0])

            if (target_rows != nrows) or (target_cols != ncols):
                # print('Number of input rows:', nrows, 'cols:', ncols)
                # print('Number of target rows:', target_rows, 'cols:', target_cols)
                not_valid = 1
                return None, None, 1

            imsize = nrows * ncols
            # offset = imsize*task_num*3 #since we are using three types of aug
            feat.extend(cls.make_features(input_color))
            target.extend(np.array(target_color).reshape(-1, ))

        return np.array(feat), np.array(target), 0

    @classmethod
    @np_cache()
    def make_features(cls, grid: np.ndarray):
        nrows, ncols = grid.shape
        features = [
            cls.make_feature(grid, i, j)
            for i,j in product(range(nrows), range(ncols))
        ]
        assert len(set(map(len,features))) == 1
        return np.array(features, dtype=np.int8)

    @classmethod
    @np_cache()
    def make_feature(cls, grid: np.ndarray, i: int, j: int) -> List:
        nrows, ncols = grid.shape
        features = [
            i, j, nrows-i, ncols-j,         # distance from edge
            i+j, i-j, j-i,                  # abs(i-j) can produce worse results
            *grid.shape, nrows*ncols,       # grid shape and pixel size
            grid[i][j],                     # grid[i][j]+1, grid[i][j]-1 = can produce worse results

            *cls.bincount(grid),
            *cls.get_moore_neighbours(grid, i, j),
            *cls.get_tl_tr(grid, i, j),

            query_not_zero(grid,i,j),
            query_max_color(grid,i,j),
            query_min_color(grid,i,j),
            query_max_color_1d(grid,i,j),
            query_min_color_1d(grid,i,j),
            query_count_colors(grid,i,j),
            query_count_colors_row(grid,i,j),
            query_count_colors_col(grid,i,j),
            query_count_squares(grid,i,j),
            query_count_squares_row(grid,i,j),
            query_count_squares_col(grid,i,j),
            max_color_1d(grid),
            min_color_1d(grid),
            get_period_length1(grid),  # has no effect
            get_period_length0(grid),  # has no effect
        ]

        neighbourhoods = [
            grid,
            cls.get_neighbourhood(grid,i,j,1),
            cls.get_neighbourhood(grid,i,j,5),
            grid[i,:], grid[:i+1,:], grid[i:,:],
            grid[:,j], grid[:,:j+1], grid[:,j:],
            grid[i:,j:], grid[:i+1,j:], grid[i:,:j+1], grid[:i+1,:j+1],
        ]
        for neighbourhood in neighbourhoods:
            features += [
                max_color(neighbourhood)           if len(neighbourhood) else 0,
                min_color(neighbourhood)           if len(neighbourhood) else 0,
                count_colors(neighbourhood)        if len(neighbourhood) else 0,
                count_squares(neighbourhood)       if len(neighbourhood) else 0,
            ]
        return features

    @classmethod
    @np_cache()
    def bincount(cls, grid: np.ndarray):
        return np.bincount(grid.flatten(), minlength=11).tolist()  # features requires a fixed length array

    @classmethod
    @np_cache()
    def get_neighbourhood(cls, grid: np.ndarray, i: int, j: int, distance=1):
        try:
            output = np.full((2*distance+1, 2*distance+1), 11)  # 11 = outside of grid pixel
            for xo, xg in enumerate(range(-distance, distance+1)):
                for yo, yg in enumerate(range(-distance, distance+1)):
                    if not 0 <= xo < grid.shape[0]: continue
                    if not 0 <= yo < grid.shape[1]: continue
                    output[xo,yo] = grid[xg,yg]
            return output
        except:
            return np.full((2*distance+1, 2*distance+1), 11)  # 11 = outside of grid pixel

    @classmethod
    @np_cache()
    def get_moore_neighbours(cls, color, cur_row, cur_col):
        nrows, ncols = color.shape
        top    = -1 if cur_row <= 0         else color[cur_row - 1][cur_col]
        bottom = -1 if cur_row >= nrows - 1 else color[cur_row + 1][cur_col]
        left   = -1 if cur_col <= 0         else color[cur_row][cur_col - 1]
        right  = -1 if cur_col >= ncols - 1 else color[cur_row][cur_col + 1]
        return top, bottom, left, right


    @classmethod
    @np_cache()
    def get_tl_tr(cls, color, cur_row, cur_col):
        nrows, ncols = color.shape
        top_left  = -1 if cur_row == 0 or cur_col == 0         else color[cur_row - 1][cur_col - 1]
        top_right = -1 if cur_row == 0 or cur_col == ncols - 1 else color[cur_row - 1][cur_col + 1]
        return top_left, top_right


class XGBSolverDart(XGBSolver):
    def __init__(self, booster='dart', **kwargs):
        self.kwargs = { "booster": booster, **kwargs }
        super().__init__(**self.kwargs)

class XGBSolverGBtree(XGBSolver):
    def __init__(self, booster='gbtree', **kwargs):
        self.kwargs = { "booster": booster, **kwargs }
        super().__init__(**self.kwargs)

class XGBSolverGBlinear(XGBSolver):
    def __init__(self, booster='gblinear', **kwargs):
        self.kwargs = { "booster": booster, "max_depth": None, **kwargs }
        super().__init__(**self.kwargs)


if __name__ == '__main__' and not settings['production']:
    solver = XGBSolver()
    solver.verbose = True
    competition = Competition()
    competition.map(solver.solve_all)
    print(competition)




### Original
# training   : {'correct': 18, 'guesses': 49, 'total': 416, 'error': 0.9567, 'time': '00:00:00', 'name': 'training'}
# evaluation : {'correct': 3, 'guesses': 19, 'total': 419, 'error': 0.9928, 'time': '00:00:00', 'name': 'evaluation'}
# test       : {'correct': 0, 'guesses': 8, 'total': 104, 'error': 1.0, 'time': '00:00:00', 'name': 'test'}

### Add i+-1, j+-1
# training   : {'correct': 50, 'total': 416, 'error': 0.8798, 'time': '00:00:00', 'name': 'training'}
# evaluation : {'correct': 19, 'total': 419, 'error': 0.9547, 'time': '00:00:00', 'name': 'evaluation'}
# test       : {'correct': 8, 'total': 104, 'error': 0.9231, 'time': '00:00:00', 'name': 'test'}
# time       : 00:00:30

### Add i+-j
# training   : {'correct': 54, 'total': 416, 'error': 0.8702, 'time': '00:00:00', 'name': 'training'}
# evaluation : {'correct': 20, 'total': 419, 'error': 0.9523, 'time': '00:00:00', 'name': 'evaluation'}
# test       : {'correct': 8, 'total': 104, 'error': 0.9231, 'time': '00:00:00', 'name': 'test'}
# time       : 00:00:28

### Add abs(i-j) - dangerous
# training   : {'correct': 58, 'total': 416, 'error': 0.8606, 'time': '00:00:00', 'name': 'training'}
# evaluation : {'correct': 17, 'total': 419, 'error': 0.9594, 'time': '00:00:00', 'name': 'evaluation'}
# test       : {'correct': 6, 'total': 104, 'error': 0.9423, 'time': '00:00:00', 'name': 'test'}
# time       : 00:00:27

### Add color+-1
# training   : {'correct': 50, 'total': 416, 'error': 0.8798, 'time': '00:00:00', 'name': 'training'}
# evaluation : {'correct': 19, 'total': 419, 'error': 0.9547, 'time': '00:00:00', 'name': 'evaluation'}
# test       : {'correct': 8, 'total': 104, 'error': 0.9231, 'time': '00:00:00', 'name': 'test'}
# time       : 00:00:31

### max_color(grid),; min_color(grid),; max_color_1d(grid),; min_color_1d(grid),; count_colors(grid),; count_squares(grid),
# training   : {'correct': 88, 'total': 416, 'error': 0.7885, 'time': '00:00:00', 'name': 'training'}
# evaluation : {'correct': 57, 'total': 419, 'error': 0.864, 'time': '00:00:00', 'name': 'evaluation'}
# test       : {'correct': 17, 'total': 104, 'error': 0.8365, 'time': '00:00:00', 'name': 'test'}
# time       : 00:00:45

### max_color(grid),; min_color(grid),; max_color_1d(grid),; min_color_1d(grid),; count_colors(grid),; count_squares(grid),
### query_not_zero(grid,i,j),; query_max_color(grid,i,j),; query_min_color(grid,i,j),; query_max_color_1d(grid,i,j),; query_min_color_1d(grid,i,j),; query_count_colors(grid,i,j),  # query_count_colors_row(grid,i,j), query_count_colors_col(grid,i,j), query_count_squares(grid,i,j), # query_count_squares_row(grid,i,j), query_count_squares_col(grid,i,j),
# training   : {'correct': 89, 'total': 416, 'error': 0.7861, 'time': '00:00:00', 'name': 'training'}
# evaluation : {'correct': 59, 'total': 419, 'error': 0.8592, 'time': '00:00:00', 'name': 'evaluation'}
# test       : {'correct': 18, 'total': 104, 'error': 0.8269, 'time': '00:00:00', 'name': 'test'}
# time       : 00:01:05

### *grid.shape
# training   : {'correct': 95, 'total': 416, 'error': 0.7716, 'time': '00:00:00', 'name': 'training'}
# evaluation : {'correct': 62, 'total': 419, 'error': 0.852, 'time': '00:00:00', 'name': 'evaluation'}
# test       : {'correct': 17, 'total': 104, 'error': 0.8365, 'time': '00:00:00', 'name': 'test'}
# time       : 00:01:18

### *np.bincount(grid.flatten(), minlength=10),; *sorted(np.bincount(grid.flatten(), minlength=10)),
# training   : {'correct': 99, 'total': 416, 'error': 0.762, 'time': '00:00:00', 'name': 'training'}
# evaluation : {'correct': 62, 'total': 419, 'error': 0.852, 'time': '00:00:00', 'name': 'evaluation'}
# test       : {'correct': 17, 'total': 104, 'error': 0.8365, 'time': '00:00:00', 'name': 'test'}
# time       : 00:01:29


### *grid.shape, nrows-i, ncols-j,
# training   : {'correct': 109, 'total': 416, 'error': 0.738, 'time': '00:00:00', 'name': 'training'}
# evaluation : {'correct': 70, 'total': 419, 'error': 0.8329, 'time': '00:00:00', 'name': 'evaluation'}
# test       : {'correct': 18, 'total': 104, 'error': 0.8269, 'time': '00:00:00', 'name': 'test'}
# time       : 00:01:21

### len(np.bincount(grid.flatten())), *np.bincount(grid.flatten(), minlength=10)
# training   : {'correct': 107, 'total': 416, 'error': 0.7428, 'time': '00:00:00', 'name': 'training'}
# evaluation : {'correct': 70, 'total': 419, 'error': 0.8329, 'time': '00:00:00', 'name': 'evaluation'}
# test       : {'correct': 18, 'total': 104, 'error': 0.8269, 'time': '00:00:00', 'name': 'test'}
# time       : 00:01:17


### neighbourhood
# training   : {'correct': 111, 'total': 416, 'error': 0.7332, 'time': '00:00:00', 'name': 'training'}
# evaluation : {'correct': 71, 'total': 419, 'error': 0.8305, 'time': '00:00:00', 'name': 'evaluation'}
# test       : {'correct': 18, 'total': 104, 'error': 0.8269, 'time': '00:00:00', 'name': 'test'}
# time       : 00:01:36

# training   : {'correct': 112, 'total': 416, 'error': 0.7308, 'time': '00:00:00', 'name': 'training'}
# evaluation : {'correct': 72, 'total': 419, 'error': 0.8282, 'time': '00:00:00', 'name': 'evaluation'}
# test       : {'correct': 19, 'total': 104, 'error': 0.8173, 'time': '00:00:00', 'name': 'test'}
# time       : 00:02:50

### without features += cls.get_neighbourhood(grid,i,j,local_neighbours).flatten().tolist()
# training   : {'correct': 112, 'total': 416, 'error': 0.7308, 'time': '00:00:00', 'name': 'training'}
# evaluation : {'correct': 78, 'total': 419, 'error': 0.8138, 'time': '00:00:00', 'name': 'evaluation'}
# test       : {'correct': 22, 'total': 104, 'error': 0.7885, 'time': '00:00:00', 'name': 'test'}
# time       : 00:02:43

### for line in ( grid[i,:], grid[:,j] ):
# training   : {'correct': 127, 'total': 416, 'error': 0.6947, 'time': '00:00:00', 'name': 'training'}
# evaluation : {'correct': 87, 'total': 419, 'error': 0.7924, 'time': '00:00:00', 'name': 'evaluation'}
# test       : {'correct': 22, 'total': 104, 'error': 0.7885, 'time': '00:00:00', 'name': 'test'}
# time       : 00:02:07

### for line_neighbourhood in [grid[i,:], grid[:i+1,:], grid[i:,:], grid[:,j], grid[:,:j+1], grid[:,j:], cls.get_neighbourhood(grid,i,j,1), cls.get_neighbourhood(grid,i,j,3), cls.get_neighbourhood(grid,i,j,5),]:
# training   : {'correct': 138, 'total': 416, 'error': 0.6683, 'time': '00:00:00', 'name': 'training'}
# evaluation : {'correct': 109, 'total': 419, 'error': 0.7399, 'time': '00:00:00', 'name': 'evaluation'}
# test       : {'correct': 31, 'total': 104, 'error': 0.7019, 'time': '00:00:00', 'name': 'test'}
# time       : 00:02:28

### for neighbourhood in [ grid[i:,j:], grid[:i+1,j:], grid[i:,:j+1], grid[:i+1,:j+1], ]
# training   : {'correct': 148, 'total': 416, 'error': 0.6442, 'time': '00:00:00', 'name': 'training'}
# evaluation : {'correct': 116, 'total': 419, 'error': 0.7232, 'time': '00:00:00', 'name': 'evaluation'}
# test       : {'correct': 33, 'total': 104, 'error': 0.6827, 'time': '00:00:00', 'name': 'test'}
# time       : 00:03:10

# XGBSolver(n_estimators=10)
# training   : {'correct': 22, 'guesses': 148, 'total': 416, 'error': 0.9471, 'time': '00:00:00', 'name': 'training'}
# evaluation : {'correct': 9, 'guesses': 116, 'total': 419, 'error': 0.9785, 'time': '00:00:00', 'name': 'evaluation'}
# test       : {'correct': 0, 'guesses': 33, 'total': 104, 'error': 1.0, 'time': '00:00:00', 'name': 'test'}
# time       : 00:00:53

# XGBSolver(n_estimators=32)
# training   : {'correct': 25, 'guesses': 255, 'total': 416, 'error': 0.9399, 'time': '00:00:00', 'name': 'training'}
# evaluation : {'correct': 10, 'guesses': 257, 'total': 419, 'error': 0.9761, 'time': '00:00:00', 'name': 'evaluation'}
# test       : {'correct': 0, 'guesses': 64, 'total': 104, 'error': 1.0, 'time': '00:00:00', 'name': 'test'}
# time       : 00:02:39

### XGBSolverDart(); XGBSolverGBtree(); XGBSolverGBlinear()
# training   : {'correct': 42, 'guesses': 254, 'total': 416, 'error': 0.899, 'time': '00:03:31', 'name': 'training'}
# evaluation : {'correct': 14, 'guesses': 242, 'total': 419, 'error': 0.9666, 'time': '00:07:17', 'name': 'evaluation'}
# test       : {'correct': 3.5, 'guesses': 61, 'total': 104, 'error': 1.0, 'time': '00:01:35', 'name': 'test'}
# time       : 00:12:23

### max_depth=10
# training   : {'correct': 43, 'guesses': 266, 'total': 416, 'error': 0.8966, 'time': '00:04:49', 'name': 'training'}
# evaluation : {'correct': 14, 'guesses': 266, 'total': 419, 'error': 0.9666, 'time': '00:08:23', 'name': 'evaluation'}
# test       : {'correct': 3.4, 'guesses': 65, 'total': 104, 'error': 1.0, 'time': '00:01:40', 'name': 'test'}
# time       : 00:14:53


#####
##### END   src_james/solver_multimodel/XGBSolver.py
#####

#####
##### START src_james/solver_multimodel/solvers.py
#####

from typing import List

# from src_james.solver_multimodel.BorderSolver import BorderSolver
# from src_james.solver_multimodel.DoNothingSolver import DoNothingSolver
# from src_james.solver_multimodel.GeometrySolver import GeometrySolver
# from src_james.solver_multimodel.GlobSolver import GlobSolver
# from src_james.solver_multimodel.SingleColorSolver import SingleColorSolver
# from src_james.solver_multimodel.Solver import Solver
# from src_james.solver_multimodel.TessellationSolver import TessellationSolver
# from src_james.solver_multimodel.XGBSolver import XGBSolver
# from src_james.solver_multimodel.XGBSolver import XGBSolverDart
# from src_james.solver_multimodel.XGBSolver import XGBSolverGBlinear
# from src_james.solver_multimodel.XGBSolver import XGBSolverGBtree
# from src_james.solver_multimodel.ZoomSolver import ZoomSolver

solvers: List[Solver] = [
    GlobSolver(),
    # DoNothingSolver(),
    # BorderSolver(),
    # GeometrySolver(),
    # SingleColorSolver(),
    # ZoomSolver(),
    # TessellationSolver(),
    # XGBSolverDart(),
    # XGBSolverGBtree(),
    # XGBSolverGBlinear(),
]


#####
##### END   src_james/solver_multimodel/solvers.py
#####

#####
##### START ./src_james/solver_multimodel/main.py
#####

import gc
import os
import time
from operator import itemgetter

# from src_james.core.DataModel import Competition
# from src_james.settings import settings
# from src_james.solver_multimodel.solvers import solvers

if __name__ == '__main__':
    print('\n','-'*20,'\n')
    print('Abstraction and Reasoning Challenge')
    print('Team: Mathematicians + Experts')
    print('https://www.kaggle.com/c/abstraction-and-reasoning-challenge')
    print('\n','-'*20,'\n')
    for solver in solvers: print(solver.__class__.__name__)
    print('\n','-'*20,'\n')

    plot_results = not settings['production']
    time_start   = time.perf_counter()
    competition  = Competition()
    scores       = { name: { solver.__class__.__name__: 0 for solver in solvers } for name in competition.keys() }
    for dataset_name, dataset in competition.items():
        time_start_dataset = time.perf_counter()
        for solver in solvers:
            print('#######', dataset_name, solver.__class__.__name__)
            if plot_results:
                scores[dataset_name][solver.__class__.__name__] += solver.plot(dataset)
            else:
                scores[dataset_name][solver.__class__.__name__] += solver.solve_all(dataset)
            # Running on Kaggle uses up nearly all 16GB of RAM
            solver.cache = {}
            gc.collect()

        dataset.time_taken = time.perf_counter() - time_start_dataset
    competition.time_taken = time.perf_counter() - time_start

    competition['test'].write_submission('submission5.csv')
    competition['test'].write_submission()

    print('-'*20)
    print('Solver Scores:')
    for dataset_name in scores.keys():
        print(f'\n# {dataset_name}')
        for solver_name, score in sorted(scores[dataset_name].items(), key=itemgetter(1), reverse=True):
            if score: print(score, solver_name)
    print('-'*20)
    print('Dataset Scores:')
    print(competition)



#####
##### END   ./src_james/solver_multimodel/main.py
#####

##### 
##### ./submission/kaggle_compile.py ./src_james/solver_multimodel/main.py
##### 
##### 2020-05-26 00:22:14+01:00
##### 
##### origin	git@github.com:seshurajup/kaggle-arc.git (fetch)
##### origin	git@github.com:seshurajup/kaggle-arc.git (push)
##### 
##### * master ccab583 [ahead 1] GlobSover | Create a lookup table of all previously seen input/output pairs
##### 
##### ccab58302b388ed65c55c9e12812429893ec29cb
##### 
