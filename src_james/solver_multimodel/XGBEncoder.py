from collections import Callable
from collections import UserList
from itertools import product
from typing import Any
from typing import Dict
from typing import List
from typing import Union

import numpy as np
from xgboost import XGBClassifier

from src_james.core.DataModel import Problem
from src_james.core.DataModel import ProblemSet
from src_james.core.DataModel import Task
from src_james.core.functions import flatten_deep
from src_james.core.functions import invoke
from src_james.solver_multimodel.Solver import Solver
from src_james.solver_multimodel.transforms.singlecolor import identity


class ProblemSetSolver(Solver):
    def solve_task(self, _task_: Task, _function_: Callable, *args, _inplace_=False, **kwargs) -> List[Any]:
        self.fit(_task_)
        if not _task_.filename in self.cache:   return []
        if self.cache[_task_.filename] is None: return []
        solutions = self.predict(_task_['test'])
        if _inplace_:
            for index in solutions['test']:
                _task_['solutions'][index] += list(solutions['test'][index])
        return solutions['test']


    def test(self, task: Task) -> bool:
        """test if the given predict correctly solves the task"""
        self.fit(task)
        if not task.filename in self.cache:   return False
        if self.cache[task.filename] is None: return False

        problemset = task['train']
        training_predictions = self.predict(problemset, task=task)
        tests_pass = len(training_predictions) == len(problemset)
        for index, prediction in enumerate(training_predictions):
            if not tests_pass: break
            if not np.array_equal( task['train'][index]['output'], prediction ):
                tests_pass = False
        return tests_pass



class XGBEncoder(ProblemSetSolver):
    dtype    = 'int8'
    encoders = {
        np.array: [identity],
    }
    features = {
        np.array:   [],
        ProblemSet: [],
        Problem:    [],
        Task:       []
    }
    # See: src_james/solver_multimodel/XGBGridSolver.hyperopt.py
    xgb_defaults = {
        'tree_method':     'exact',
        'eval_metric':     'error',
        'sampling_method': 'uniform',
        # 'min_child_weight': 0,
        # 'max_depth':        1,
        'objective':       'reg:squarederror',
        'max_delta_step':   1,   # makes the rules more precise
        # 'num_classes':     11,
        # 'n_estimators':    16,
        # 'max_depth':       1,
    }

    def __init__(self,
                 input_encoder:  Callable = None,
                 output_encoder: Callable = None,
                 features:       Dict = None,
                 xgb_args:       Dict = {}
    ):
        super().__init__()
        self.input_encoder  = input_encoder
        self.output_encoder = output_encoder
        self.features       = features if features is not None else self.__class__.features
        self.xgb_args       = { **self.xgb_defaults, **xgb_args }
        # self.onehot         = OneHotEncoder(handle_unknown='ignore').fit([list(range(11+1))])

    def __call__(self, problemset: ProblemSet, task: Task):
        return self.predict(problemset=problemset, task=task, **self._chained_args)


    def create_xgb(self):
        xgb = XGBClassifier(n_jobs=-1, **self.xgb_args)
        return xgb


    def fit(self, task: Task) -> bool:
        """Find the best input_encoder/output_encodr for the task """
        if task.filename in self.cache: return True

        problemset      = task['train']
        input_encoders  = (self.input_encoder, ) if self.input_encoder  else self.encoders[np.array]
        output_encoders = (self.output_encoder,) if self.output_encoder else self.encoders[np.array]
        for input_encoder, output_encoder in product(input_encoders, output_encoders):
            if not callable(input_encoder):  continue
            if not callable(output_encoder): continue
            self.input_encoder  = input_encoder   # this is idempotent for a one element list
            self.output_encoder = output_encoder  # cache the last value in this loop

            inputs  = self.generate_input_array(  problemset )
            outputs = self.generate_output_array( problemset )

            # See: venv/lib/python3.6/site-packages/xgboost/sklearn.py:762
            xgb = self.create_xgb()
            xgb.fit(inputs, outputs, verbose=False)

            self.cache[task.filename] = (xgb,)  # Needs to be set for self.test() to read and prevent an infinite loop
            self._chained_args = { "task": task }
            if self.test(task):
                return True
        else:
            self.cache[task.filename] = None
            return False


    def predict(self,
                problemset: Union[ProblemSet, Task],
                xgb:   XGBClassifier = None,
                task:  Task = None,
                *args, **kwargs
    ) -> Any:
        if isinstance(problemset, Task):      problemset = problemset['test']
        task = problemset.task
        if task.filename not in self.cache:   self.fit(task)
        if self.cache[task.filename] is None: return None  # Unsolvable mapping

        task: Task          = task or self._chained_args.get('task')
        xgb:  XGBClassifier = xgb  or self.cache[task.filename][0]

        input  = self.generate_input_array(problemset)
        output = xgb.predict(input)
        return output


    def test(self, task: Task) -> bool:
        """test if the given predict correctly solves the task"""
        if task.filename not in self.cache:             self.fit(task)
        if self.cache.get(task.filename, True) is None: return False

        problemset = task['test']
        expected = self.generate_output_array(problemset)
        actual   = self.predict(problemset)
        is_valid = np.array_equal(expected, actual)
        print('test | expected: ', expected, ' | actual: ', actual, ' | is_valid: ', is_valid)
        return is_valid


    def onehotencode(self, input, maxsize=11):
        output = []
        for item in input:
            if isinstance(item, (list, UserList, np.ndarray)):
                item = self.onehotencode(item, maxsize)
            value = int(item)
            encoded = np.zeros(maxsize, dtype=np.int8)
            encoded[value] = 1
            output.append(encoded)
        return np.array(output)



    # @np_cache()
    def generate_output_array(self, problemset: ProblemSet, output_encoder=None):
        output_encoder = output_encoder or self.output_encoder
        outputs = []
        for problem in problemset:
            if problem['output'] is None: continue
            input  = problem['input']
            output = problem['output']
            if callable(output_encoder):
                encoded = output_encoder(output)
                # encoded = input.flatten() + encoded  # XGBoost can't predict unseen input numbers
                # encoded   = self.onehotencode(output)
                outputs.append(encoded)
        return np.array(outputs, dtype=np.int8).flatten()


    # @np_cache()
    def generate_input_array(self, problemset: ProblemSet, input_encoder=None) -> np.ndarray:
        mappings = self.generate_input_mappings(problemset, self.features)
        for index, mapping in enumerate(mappings):
            # noinspection PyTypeChecker
            mappings[index] = flatten_deep(mapping.values())
        mappings_array = np.array(mappings, dtype=np.int8)  # dtype=np.int8 is broken
        assert mappings_array.shape[0] == len(problemset)
        return mappings_array

    # @np_cache()
    def generate_input_mappings(self, problemset: ProblemSet, input_encoder=None) -> List[Dict[Callable, Any]]:
        # XGBoost requires a 2D array, one slice for each problem
        input_encoder = input_encoder or self.input_encoder
        mappings  = []
        for problem in problemset:
            mapping = {}
            for feature_fn in self.features[Task]:
                mapping[feature_fn] = invoke(feature_fn, problemset.task)
            for feature_fn in self.features[ProblemSet]:
                mapping[feature_fn] = invoke(feature_fn, problemset)
            for feature_fn in self.features[Problem]:
                mapping[feature_fn] = invoke(feature_fn, problem)
            for feature_fn in self.features[np.array]:
                input               = input_encoder(problem['input']) if callable(input_encoder) else problem['input']
                mapping[feature_fn] = invoke(feature_fn, input)
            mappings.append(mapping)
        return mappings




