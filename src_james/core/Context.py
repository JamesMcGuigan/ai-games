from collections import UserDict

import numpy as np

from src_james.core.DataModel import Problem



class Context(UserDict):
    def __init__( self, problem: Problem, *args, **kwargs ):
        super().__init__(**kwargs)
        if len(args) == 0: args = [ problem['input'] ]
        arg_dict   = { str(index): arg for index,arg in enumerate(args) }
        self.data  = {
            **arg_dict,
            "input":      problem['input'],
            "problem":    problem,
            "problemset": problem.problemset,
            "task":       problem.task,
            **kwargs,
            }

    def __str__(self):
        output = {}
        for key, value in self.data.items():
            if   isinstance(value, np.ndarray):        output[key] = np.array(value.shape)
            elif hasattr(self.data[key], '__dict__'):  output[key] = type(value)
            else:                                      output[key] = value
        return str(output)
