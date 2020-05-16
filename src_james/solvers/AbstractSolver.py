import inspect
import traceback
from collections import UserDict, defaultdict
from itertools import chain, product
from pprint import pprint
from typing import Any, Callable, DefaultDict, Dict, List, Tuple, Type, Union

import numpy as np
from sympy import Symbol, symbols

from src_james.DataModel import Competition, Hashed, Problem, Task
from src_james.heuristics.Queries import Query
from src_james.settings import settings
from src_james.solvers.OutputGridSolver import OutputGridSizeTransforms



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


# class RuleSet(UserList):
#     def __init__( self, *args ):
#         for rule in args: assert isinstance(rule, Rule)
#         super().__init__(*args)
#
#     def __call__(self, input: np.ndarray, *args, **kwargs):
#         output = np.copy(input)
#         for rule in self.data:
#             output = rule(output, *args, **kwargs)
#         return output


class Rule(object):
    def __init__(self, function: Callable, arguments={}):
        self.function  = function
        self.arguments = arguments
        # self.context   = context


    @classmethod
    def kwargs_from_context( cls, function, context: Context, arguments={}):
        signature = inspect.signature(function)
        kwargs    = {}
        for key, parameter in signature.parameters.items():
            if key in arguments.keys():
                argument = arguments[key]

                # Resolve argument symbols from context
                if isinstance(argument, Symbol):
                    name = argument.name
                    if name in context:
                        argument = context[name]

                # Resolve arguments[key]
                if AbstractSolver.isinstance(argument, parameter.annotation):
                    if callable(argument):
                        kwargs[key] = cls.call_with_context(argument, context)
                    else:
                        kwargs[key] = argument
                    continue

        # See if we can typematch from context - but only if we get a single typematch
        for key, parameter in signature.parameters.items():
            if key in kwargs: continue  # already solved
            options = []
            for name in context.keys():
                if AbstractSolver.isinstance(context[name], parameter.annotation):
                    if context[name] not in options:
                        options.append( context[name] )
            if len(options) == 1:
                kwargs[key] = list(options)[0]

        for key, parameter in signature.parameters.items():
            if not key in kwargs and parameter.default is parameter.empty:
                if settings['debug']:
                    print(f'{cls.__name__}.kwargs_from_context() - unable to resolve | {function} {signature} | using: arguments={arguments}, context={context}')
                return None

        return kwargs


    @classmethod
    def call_with_context(cls, function, context: Context, arguments: Dict={} ) -> Any:
        kwargs = cls.kwargs_from_context(function, context, arguments)
        if kwargs is None: return None
        try:
            output = function(**kwargs)
            return output
        except TypeError as exception:
            #kwargs = cls.kwargs_from_context(function, context, arguments)
            pass   # kwargs didn't resolve all required arguments - ignore
        except Exception as exception:
            #kwargs = cls.kwargs_from_context(function, context, arguments)
            if settings['debug']:
                print('-'*20)
                print(f"Exception: {cls.__name__}.call_with_context()")
                print(f"function={function}")
                print(f"kwargs={kwargs}")
                #print(f"arguments={arguments}")
                #print(f"context={context}")
                traceback.print_exception(type(exception), exception, exception.__traceback__)
                print('-'*20)
            return None


    def __call__(self, context: Union[Problem,Context]) -> Any:
        if not isinstance(context, Context): context = Context(context)
        output  = self.call_with_context(self.function, context, self.arguments)
        return output

    def __repr__(self):
        arguments = { key: value.__name__ if hasattr(value, '__name__') else str(value)
                      for key, value in self.arguments.items() }
        arguments = " ".join( f"{k}={v}" for k,v in arguments.items() )
        return f'<Rule {self.function.__name__}({arguments})>'

# class RuleSetSolver(object):
#     def __init__(self,
#                  name:       str                = '',
#                  preprocess: Callable           = None,
#                  functions:  List[Callable]     = [],
#                  arguments:  List[Any,Callable] = [],
#     ):
#         self.name       = name
#         self.preprocess = deepcopy(preprocess)  or self.preprocess
#         self.functions  = deepcopy(functions)   or self.functions
#         self.arguments  = deepcopy(arguments)   or self.arguments
#
#     def solve( self, inputs: List[Any], outputs: List[Any], context={} ):
#         pass




class AbstractSolver(Hashed):
    functions = []
    arguments = []


    def __init__( self ):
        super().__init__()
        self.cache: Dict[str,Rule] = {}


    def preprocess( self, input: np.ndarray ) -> Any:
        return input


    # def create_context( self, problem: Problem, *args, **kwargs ) -> Dict[Union[str,int], Any]:
    #     arg_dict = { str(index): arg for index,arg in enumerate(args) }
    #     context  = {
    #         **arg_dict,
    #         "input":      problem['input'],
    #         "problem":    problem,
    #         "problemset": problem.problemset,
    #         "task":       problem.task,
    #         **kwargs,
    #     }
    #     return context


    def solve_one( self, task: Task, context={} ) -> Union[Rule,None]:
        rules = self.solve(task, context, max_solutions=1)
        return rules[0] if len(rules) else None

    def solve( self, task: Task, context={}, max_solutions=np.inf ) -> List[Rule]:
        problemset = task['train']
        inputs     = [ self.preprocess(problem) for problem in problemset.inputs  ]
        outputs    = [ self.preprocess(problem) for problem in problemset.outputs ]
        assert len(inputs)
        assert len(inputs) == len(outputs)

        valid_rules = []
        context = Context(problemset[0], inputs[0])
        for function in self.functions:
            argument_permutations = self.argument_permutations(function, context, self.arguments)
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

    @classmethod
    def group_context_by_type( cls, context: Union[Dict,UserDict] ) -> DefaultDict[Type,List[Any]]:
        grouped = defaultdict(list)
        for name, item in context.items():
            types = cls.types(item)
            for type in types:
                if item in grouped[type]: continue
                grouped[type].append( symbols(name) )
        return grouped

    @classmethod
    def group_by_type( cls, collection: List[Any] ) -> DefaultDict[Type,List[Any]]:
        grouped = defaultdict(list)
        for item in collection:
            types = cls.types(item)
            for type in types:
                if item in grouped[type]: continue
                grouped[type].append(item)
        return grouped


    @classmethod
    def types( cls, input: Any ) -> Tuple[Type]:
        """as cls.type() but always return a iterable tuple"""
        types = cls.type(input)
        if not isinstance(types, tuple): types = (types,)
        return types


    # noinspection PyTypeChecker
    @classmethod
    def type( cls, input: Any ) -> Union[Type, Tuple[Type]]:
        # https://stackoverflow.com/questions/45957615/check-a-variable-against-union-type-at-runtime-in-python-3-6
        if isinstance(input, Type):       return input
        if hasattr(input, '__origin__'):
            if input.__origin__ == Union: return tuple( cls.type(arg) for arg in input.__args__ ) # Union[int,str] -> (<class 'int'>, <class 'str'>)
            else:                         return input.__origin__  # Tuple[int,int] -> <class 'tuple'>
        if callable(input):               return cls.type( inspect.signature(input).return_annotation )
        else:                             return type(input)


    @classmethod
    def isinstance( cls, a, b ):
        a_types = cls.types(a)
        b_types = cls.types(b)
        return any([ a_type == b_type for a_type, b_type in product(a_types, b_types) ])


    @classmethod
    def argument_permutations(cls, function: Callable, context: Context, arguments=[]):
        arg_options = defaultdict(list)
        parameters  = inspect.signature(function).parameters
        if len(parameters) == 0: return []

        arguments_by_type = cls.group_by_type(arguments)
        context_by_type   = cls.group_context_by_type(context)
        for index, (key, parameter) in enumerate(parameters.items()):
            index      = str(index)  # create_context() stores numbers as strings, and symbols() requires a string
            annotation = parameters[key].annotation
            for annotation_type in cls.types(annotation):

                # Type input as first parameter
                if index in context:
                    if cls.isinstance(context[index], annotation_type):
                        arg_options[key].append( symbols(index) )
                        continue

                # Add from context by type
                for item in context_by_type[ annotation_type ]:
                    if item not in arg_options[key]:  # not everything hashes when using a set
                        arg_options[key].append(item)

                # Add from self.arguments by type
                for item in arguments_by_type[ annotation_type ]:
                    if item not in arg_options[key]:
                        arg_options[key].append(item)

        # https://stackoverflow.com/questions/15211568/combine-python-dictionary-permutations-into-list-of-dictionaries
        permutations = [ dict(zip(arg_options, options)) for options in product(*arg_options.values()) ]
        return permutations


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
