from collections import UserDict
from typing import Any
from typing import Dict
from typing import List
from typing import Union

import numpy as np


# noinspection PyUnresolvedReferences
class Problem(UserDict):
    """ Problem: An input + output Grid pair """
    dtype = np.int8
    def __init__(self, problem: Union[Dict[str,np.ndarray],'Problem'], problemset: 'ProblemSet'):
        super().__init__()
        self._hash = 0
        self.problemset: 'ProblemSet'         = problemset
        self.task:       'Task'               = problemset.task
        self.raw:        Dict[str,np.ndarray] = problem.raw if isinstance(problem, Problem) else problem

        self.data = {}
        for key in ['input', 'output']:
            value = self.cast(problem.get(key, None))
            self.data[key] = value

    def cast(self, value: Any):
        if value is None: return None
        value = np.array(value, dtype=self.dtype)
        # value = np.ascontiguousarray(value, dtype=self.dtype)  # disable: could potntually mess with hashing
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
