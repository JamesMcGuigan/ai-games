from collections import Callable
from typing import Any
from typing import Dict
from typing import List
from typing import Union

import numpy as np
from itertools import product
from xgboost import XGBClassifier

from src.datamodel.Problem import Problem
from src.datamodel.ProblemSet import ProblemSet
from src.datamodel.Task import Task
from src.solver_multimodel.core.ProblemSetSolver import ProblemSetSolver
from src.solver_multimodel.functions.transforms.singlecolor import identity
from src.util.functions import flatten_deep
from src.util.functions import invoke


class ProblemSetEncoder(ProblemSetSolver):
    debug    = False,
    encoders = {
        np.array: [identity],
    }
    features = {
        np.array:   [],
        ProblemSet: [],
        Problem:    [],
        Task:       []
    }
    encoder_defaults = {}

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
        self.encoder_args   = {**self.encoder_defaults, **xgb_args}


    def __call__(self, problemset: ProblemSet, task: Task):
        return self.predict(problemset=problemset, task=task, **self._chained_args)


    def create_encoder(self):
        """Return an Encoder that implements .fit() and .predict()"""
        raise NotImplementedError


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
            encoder = self.create_encoder()
            encoder.fit(inputs, outputs, verbose=False)

            self.cache[task.filename] = (encoder,)  # Needs to be set for self.test() to read and prevent an infinite loop
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
        task       = task or (problemset if isinstance(problemset, Task) else problemset.task)
        problemset = (problemset['test'] if isinstance(problemset, Task) else problemset )
        if task.filename not in self.cache:   self.fit(task)
        if self.cache[task.filename] is None: return None  # Unsolvable mapping

        task: Task          = task or self._chained_args.get('task')
        xgb:  XGBClassifier = xgb  or self.cache[task.filename][0]

        input  = self.generate_input_array(problemset)
        output = xgb.predict(input)
        return output


    def test(self, task: Task) -> bool:
        """test if .predict() correctly solves the task"""
        if task.filename not in self.cache:             self.fit(task)
        if self.cache.get(task.filename, True) is None: return False

        train_problemset = task['train']  # we test on the train side, to validate if we can .predict() on test
        train_expected   = self.generate_output_array(train_problemset)
        train_actual     = self.predict(train_problemset)
        train_valid      = np.array_equal(train_expected, train_actual)

        if self.debug:
            test_problemset = task['test']  # we test on the train side, to validate if we can .predict() on test
            test_expected = self.generate_output_array(test_problemset)
            test_actual   = self.predict(test_problemset)
            test_valid = np.array_equal(test_expected, test_actual)
            print(" | ".join([
                task.filename.ljust(24),
                f'{str(train_valid).ljust(5)} -> {str(test_valid).ljust(5)}',
                f'{train_expected} -> {train_actual}',
                f'{test_expected} -> {test_actual}',
            ]))

        return train_valid


    # def onehotencode(self, input, maxsize=11):
    #     output = []
    #     for item in input:
    #         if isinstance(item, (list, UserList, np.ndarray)):
    #             item = self.onehotencode(item, maxsize)
    #         value = int(item)
    #         encoded = np.zeros(maxsize, dtype=np.int8)
    #         encoded[value] = 1
    #         output.append(encoded)
    #     return np.array(output)


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


