from collections import UserDict, defaultdict
from itertools import chain
from pprint import pprint
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np

from src_james.DataModel import Competition, Hashed, Problem, Task
from src_james.heuristics.Queries import Query
from src_james.solvers.Context import Context
from src_james.solvers.Rule import Rule
from src_james.solvers.old.OutputGridSolver import OutputGridSizeTransforms



class AbstractSolver(Hashed):
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
                    context  = Context(problemset[index], input)   # TODO: create context class
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
        rule = self.cache.get(task.filename, None)
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



class AbstractOutputGridSizeSolver(AbstractSolver):
    dtype: Tuple[int,int]
    functions = [
        OutputGridSizeTransforms.identity,
        OutputGridSizeTransforms.fixed_size,
        OutputGridSizeTransforms.ratio,
    ]
    arguments = [
        Query.grid_size_ratio_task,
        Query.count_nonzero,
        Query.unique_colors,
        1/4, 1/3, 1/2, 1, 2, 3, 4,
    ]
    def preprocess( self, input: np.ndarray ) -> Any:
        return input.shape


if __name__ == '__main__':
    solver = AbstractOutputGridSizeSolver()
    task   = Task('evaluation/68b67ca3.json')
    rule   = solver.solve(task)
    # assert rule.name == 'GridSizeIntegerMultiple'
    # assert rule.args == [(2.0, 2.0)]

    competition  = Competition()
    solutions    = defaultdict(list)
    solved_files = { name: [] for name in competition.keys() }
    error_files  = { name: [] for name in competition.keys() }
    counts       = defaultdict(int)
    for name, dataset in competition.items():
        for task in dataset:
            #rule  = solver.solve_one(task)
            rules = solver.solve(task)
            if not len(rules):
                error_files[name].append(task.filename)
            else:
                solutions[task.filename] += rules
                solved_files[name].append(task.filename)
                counts[name] += 1

    print()
    print('Counts')
    pprint(counts)

    print('Solutions')
    for filename, matching_rules in solutions.items():
        if len(matching_rules):
            print(filename)
            for rule in matching_rules: print(rule)
            print()

    print()
    print('Errors')
    for filename in chain(*error_files.values()):
        print(filename)
