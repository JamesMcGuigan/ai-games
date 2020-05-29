import inspect
import pprint
from collections import defaultdict
from typing import Any
from typing import Callable
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import pydash as py

from src.core.DataModel import Competition
from src.core.DataModel import ProblemSet
from src.core.DataModel import Task
from src.heuristics.Queries import Query
from src.settings import settings


# from pydash import group_by

# NOTE: This is getting half way to the level of abstraction required, but we need to go to the next level of inception
class Rule:
    def __init__(self,
                 function:  Callable,
                 args:      Union[list,dict]            = [],
                 input_transform:  Callable[[np.ndarray], Any] = None,
                 output_transform: Callable[[np.ndarray], Any] = None,
                 name:      str                         = ''
    ):
        assert callable(function)
        self.function  = function
        self.args      = args
        self.input_transform  = input_transform
        self.output_transform = output_transform
        self.name      = name or self.__class__.__name__

    def test(self, input: np.ndarray, output: np.ndarray):
        guess = self.__call__(input)
        return np.equal(guess, output)

    def __call__(self, input: Any):
        if callable(self.input_transform):
            input = self.input_transform(input)

        if isinstance(self.args, dict):
            args   = { key: value(input) if callable(value) else value for key,value in self.args.items()  }
            output = self.function(input, **args)
        else:
            args   = [ value(input) if callable(value) else value for value in self.args ]
            output = self.function(input, *args)

        if callable(self.output_transform):
            output = self.output_transform(output)
        return output


    def __repr__(self):
        return self.__str__()
    def __str__(self):
        return f"{self.name or self.__class__.__name__ }({self.args or '' })"
    def __hash__(self):
        return hash(( hash(value) for value in [self.function, self.args, self.input_transform, self.output_transform, self.name] ))



class OutputGridSizeTransforms:
    @classmethod
    def identity( cls, input: Tuple[int, int] ) -> Tuple[int, int]:
        return input

    @classmethod
    def fixed_size(cls, size: Union[int, Tuple[int,int]]) -> Tuple[int,int]:
        if isinstance(size, (int,float)):
            size = (int(size), int(size))
        return size

    @classmethod
    def ratio(cls, input: Union[np.ndarray,Tuple[int,int]], ratio: Union[int, float, Tuple[int,int]]) -> Tuple[int,int]:
        if ratio is None:                   return (0,0)
        if isinstance(input, np.ndarray):   input = input.shape
        if isinstance(ratio, (int,float)):  ratio = (ratio, ratio)
        try:
            return (
                int(input[0] * ratio[0]),
                int(input[1] * ratio[1]),
            )
        except:
            return (0,0)




class OutputGridSizeSolver:
    cache = {}
    dtype: Tuple[int,int].__origin__
    functions = [
        OutputGridSizeTransforms.identity,
        OutputGridSizeTransforms.fixed_size,
        OutputGridSizeTransforms.ratio,
    ]
    arguments = py.group_by([
        Query.grid_size_ratio_task,
        Query.count_nonzero,
        Query.unique_colors
    ], lambda f: inspect.signature(f).return_annotation)

    def __init__(self):
        pass

    @classmethod
    def equals(cls, input: Tuple[int,int], output: Tuple[int,int]):
        return input == output

    def fit_predict(self, task: Task):
        rule    = self.fit(task['train'])
        predict = self.predict(task['test'])
        return predict

    def fit(self, problemset: Union[ProblemSet,Task]):
        if isinstance(problemset, Task): return self.fit(problemset['train'])

        rule   = None
        ratios = list({
            (
                problem['input'].shape[0] / problem['output'].shape[0],
                problem['input'].shape[1] / problem['output'].shape[1],
            )
            for problem in problemset
        })
        if len(ratios) == 1:
            ratio = ratios[0]
            if ratio == (1,1):
                rule = Rule(lambda input: input,
                            input_transform=self.transform,
                            name='GridSizeSame')

            elif ( float(ratio[0]).is_integer() and float(ratio[1]).is_integer() ):
                rule  = Rule(lambda input, ratio: (input[0]*ratio[0],input[1]*ratio[1]),
                             args=[ ratio ],
                             input_transform=self.transform,
                             name='GridSizeIntegerMultiple')

            elif float(1/ratio[0]).is_integer() and float(1/ratio[1]).is_integer():
                rule  = Rule(lambda input, ratio: (input[0]*ratio[0],input[1]*ratio[1]),
                             args=[ ratio ],
                             input_transform=self.transform,
                             name='GridSizeIntegerDivisor')
            else:
                rule  = Rule(lambda input, ratio: (input[0]*ratio[0],input[1]*ratio[1]),
                             args=[ ratio ],
                             input_transform=self.transform,
                             name='GridSizeInputRatio')
        else:
            output_shapes = list({ problem['output'].shape for problem in problemset })
            if len(output_shapes) == 1:
                rule = Rule(lambda input, output_shape: output_shape,
                            args=[ output_shapes[0] ],
                            input_transform=self.transform,
                            name='GridSizeFixedSizeOutput')

        self.cache[problemset.filename] = rule
        return rule

    def transform( self, input: np.ndarray ):
        return input.shape

    def test(self, problemset: ProblemSet, rule: Rule = None):
        if isinstance(problemset, Task): return self.fit(problemset['train'])
        if not rule: rule = self.cache[problemset.filename]
        for problem in problemset:
            input  = self.transform(problem['input'])
            output = self.transform(problem['output'])
            guess  = rule(input)
            if not np.equal(guess, output):
                return False
        return True


    def predict(self, problemset: Union[ProblemSet,Task], rule: Rule=None, *args, task: Task=None, **kwargs) -> Union[None,List[np.ndarray]]:
        task       = task or (problemset if isinstance(problemset, Task) else problemset.task)
        problemset = (problemset['test'] if isinstance(problemset, Task) else problemset )
        if task.filename not in self.cache:   self.fit(task)
        if self.cache[task.filename] is None: return None  # Unsolvable mapping

        if not rule: rule = self.cache[problemset.filename]
        if not callable(rule): return None
        return [ rule(problem['input']) for problem in problemset ]



if __name__ == '__main__' and not settings['production']:
    solver      = OutputGridSizeSolver()

    task = Task('evaluation/68b67ca3.json')
    rule = solver.fit(task)
    assert rule.name == 'GridSizeIntegerMultiple'
    assert rule.args == [(2.0, 2.0)]


    competition = Competition()
    files  = { name: defaultdict(list) for name in competition.keys() }
    counts = { name: defaultdict(int)  for name in competition.keys() }
    for name, dataset in competition.items():
        for task in dataset:
            rule = solver.fit(task)
            files[name][rule and rule.name].append(task.filename)
            counts[name][rule and rule.name] += 1

    print('Counts')
    pprint.pprint(counts)

    print('Errors')
    for name in files.keys():
        print(name, files[name][None])
