from itertools import combinations, product

import numpy as np

from src_james.original.Solver import Solver



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
        if task['file'] in self.cache: return True

        max_roll = (self.task_grid_max_dim(task) + 1) // 2
        for key, (function, arglist) in self.actions.items():
            if function == np.roll: arglist = product(range(-max_roll,max_roll),[0,1])
            for args in arglist:
                if self.is_lambda_valid(task, function, *args):
                    self.cache[task['file']] = (function, args)
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
                    self.cache[task['file']] = (function, (args1,args2))
                    if self.verbose: print(function, (args1,args2))
                    return True
        return False

    def action(self, grid, function=None, args=None, task=None):
        try:
            return function(grid, *args)
        except Exception as exception:
            if self.verbose: print(function, args, exception)
            return grid
