from collections import UserList
from typing import Dict
from typing import List
from typing import Union

import numpy as np

from src.datamodel.Problem import Problem



# noinspection PyUnresolvedReferences
class ProblemSet(UserList):
    """ ProblemSet: An array of either test or training Problems """
    _instance_count = 0

    # def __new__(cls, input_outputs: Union[List[Dict[str, np.ndarray]],ProblemSet], *args, **kwargs):
    #     if isinstance(input_outputs, ProblemSet): return input_outputs
    #     else:                                     return super(ProblemSet, cls).__new__(cls, *args, **kwargs)

    def __init__(self,
                 input_outputs: Union[List[Dict[str, np.ndarray]],List[Problem],'ProblemSet'],
                 test_or_train: str,
                 task: 'Task'
                 ):
        super().__init__()
        self.task:          'Task'                       = task
        self.test_or_train: str                        = test_or_train
        self.raw:           List[Dict[str,np.ndarray]] = input_outputs.raw if isinstance(input_outputs, ProblemSet) else input_outputs
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
