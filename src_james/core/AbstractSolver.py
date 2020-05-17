from collections import UserDict
from typing import Any, Callable, Dict, List, Union

import numpy as np

from src_james.core.Context import Context
from src_james.core.DataModel import Problem, Task
from src_james.core.Rule import Rule



class AbstractSolver:
    functions = []
    arguments = []

    def __init__( self ):
        super().__init__()
        self.cache: Dict[str,Rule] = {}


    def preprocess( self, input: np.ndarray ) -> Any:
        return input


    def solve_one( self, task: Task, context={} ) -> Union[Rule,None]:
        rules = self.solve(task, context, max_solutions=1)
        return rules[0] if len(rules) else None


    def solve( self, task: Task, context: Union[Dict,UserDict]={}, max_solutions=np.inf ) -> List[Rule]:
        problemset = task['train']
        inputs     = [ self.preprocess(problem) for problem in problemset.inputs  ]
        outputs    = [ self.preprocess(problem) for problem in problemset.outputs ]
        assert len(inputs)
        assert len(inputs) == len(outputs)

        valid_rules = []
        context = Context(problemset[0], inputs[0], **context)
        for function in self.functions:
            argument_permutations = Rule.argument_permutations(function, context, self.arguments)
            for arguments in argument_permutations:
                rule = Rule(function, arguments)
                rule_is_valid = True
                for index in range(len(inputs)):
                    input    = inputs[index]
                    context  = Context(problemset[index], input)
                    actual   = rule.__call__( context=context )
                    expected = outputs[index]
                    if not np.array_equal(actual, expected):
                        rule_is_valid = False
                        break
                if rule_is_valid:
                    valid_rules.append(rule)
                    # Only need to check this when len(valid_rules) has changed
                    if max_solutions and max_solutions <= len(valid_rules):
                        return valid_rules
        return valid_rules


    def predict( self, problem: Problem ) -> Any:
        """return the predicted problem output"""
        rule = self.cache.get(problem.filename, None)
        assert problem.filename in self.cache
        assert callable(rule)

        result  = rule(problem['input'])
        return result


    def test( self, task: Task, rule: Union[Rule,Callable]=None ) -> bool:
        """test if the cached rule solves all the task['train'] examples"""
        rule = rule or self.cache.get(task.filename, None)
        assert callable(rule)

        is_valid = True
        for problem in task['train']:
            prediction = rule(problem['input'])
            actual     = problem['output']
            if np.array_equal(prediction, actual):
                is_valid = False
                break
        return is_valid



