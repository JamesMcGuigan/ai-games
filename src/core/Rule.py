import inspect
import traceback
from collections import defaultdict
from collections import UserDict
from functools import lru_cache
from itertools import product
from typing import Any
from typing import Callable
from typing import DefaultDict
from typing import Dict
from typing import List
from typing import Set
from typing import Tuple
from typing import Type
from typing import Union

from src.core.Context import Context
from src.core.DataModel import Problem
from src.core.Symbol import Symbol
from src.settings import settings


class Rule(object):
    def __init__(self, function: Callable, arguments: Dict = None):
        self.function  = function
        self.arguments = arguments or dict()
        self._hash     = hash(self.function) + hash(tuple(self.arguments.items()))


    def __call__(self, context: Union[Problem,Context]) -> Any:
        if not isinstance(context, Context): context = Context(context)
        output  = self.call_with_context(self.function, context, self.arguments)
        return output

    def __repr__(self):
        arguments = { key: value.__name__ if hasattr(value, '__name__') else str(value)
                      for key, value in self.arguments.items() }
        arguments = " ".join( f"{k}={v}" for k,v in arguments.items() )
        return f'<Rule {self.function.__name__}({arguments})>'

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if not isinstance(other, Rule):        return False
        if self.function  != other.function:   return False
        if self.arguments != other.arguments:  return False   # compare by value - https://stackoverflow.com/questions/4527942/comparing-two-dictionaries-and-checking-how-many-key-value-pairs-are-equal
        return True

    @classmethod
    def kwargs_from_context( cls, function, context: Context, arguments={}, strict=False):
        signature = inspect.signature(function)
        kwargs    = {}
        for key, parameter in signature.parameters.items():
            if key not in arguments.keys(): continue
            argument = arguments[key]

            # Resolve argument symbols from context
            if isinstance(argument, Symbol):
                name = argument.name
                if name in context:
                    argument = context[name]

            # Resolve arguments[key]
            if cls.isinstance(argument, parameter.annotation):
                if callable(argument):
                    kwargs[key] = cls.call_with_context(argument, context)
                else:
                    kwargs[key] = argument
                continue

        # See if we can typematch from context - strict means enforcing a unique typematch
        seen = set()
        context_by_type = cls.group_context_by_type(context)
        for key, parameter in signature.parameters.items():
            if key in kwargs:                                              continue  # already solved
            if parameter.annotation not in context_by_type:                continue
            if strict and len(context_by_type[parameter.annotation]) != 1: continue
            for symbol in context_by_type[parameter.annotation]:
                if symbol in seen:             continue
                if symbol.name not in context: continue
                seen.add(symbol)
                kwargs[key] = context[symbol.name]
                break


        for key, parameter in signature.parameters.items():
            if not key in kwargs and parameter.default is parameter.empty:
                if settings['debug']:
                    print(f'{cls.__name__}.kwargs_from_context() - unable to resolve | {function} {signature} | using: arguments={arguments}, context={context}')
                return None

        return kwargs


    @classmethod
    def call_with_context(cls, function, context: Context, arguments: Dict={} ) -> Any:
        kwargs = cls.kwargs_from_context(function, context, arguments)
        output = None
        try:
            if kwargs is not None:
                output = function(**kwargs)
        except TypeError as exception:
            if settings['debug']: cls.print_exception(exception, function, kwargs, arguments, context)
        except Exception as exception:
            if settings['debug']: cls.print_exception(exception, function, kwargs, arguments, context)

        if output is None and settings['debug']:
            kwargs = cls.kwargs_from_context(function, context, arguments)
        return output


    @classmethod
    def print_exception(cls, exception, function=None, kwargs=None, arguments=None, context=None ):
        if settings['debug']:
            print('-'*20)
            print(f"Exception: {cls.__name__}.call_with_context()")
            print(f"function={function}")
            print(f"kwargs={kwargs}")
            #print(f"arguments={arguments}")
            #print(f"context={context}")
            traceback.print_exception(type(exception), exception, exception.__traceback__)
            print('-'*20)


    @classmethod
    @lru_cache(1024)  # Profiler: expensive function
    def group_context_by_type( cls, context: Union[Dict,UserDict] ) -> DefaultDict[Type,List[Any]]:
        grouped = defaultdict(list)
        for name, item in context.items():
            types = cls.types(item)
            for type in types:
                grouped[type].append( Symbol(name) )
        return grouped


    @classmethod
    def group_by_type( cls, collection: List[Any] ) -> DefaultDict[Type,Set[Any]]:
        grouped = defaultdict(set)
        for item in collection:
            types = cls.types(item)
            for type in types:
                grouped[type].add(item)
        return grouped


    @classmethod
    def types( cls, input: Any ) -> Tuple[Type]:
        """as cls.type() but always return a iterable tuple"""
        types = cls.type(input)
        if not isinstance(types, tuple): types = (types,)
        return types



    @classmethod
    def type( cls, input: Any ) -> Union[Type, Tuple[Type]]:
        # Profiler: was 20% now 13% of runtime - selectively cache functions
        if callable(input):
            hkey = hash(input)
            if hkey not in cls._type_cache:
                cls._type_cache[hkey] = cls._type(input)
            return cls._type_cache[hkey]
        else:
            return cls._type(input)


    _type_cache = {}


    # noinspection PyTypeChecker
    @classmethod
    def _type( cls, input: Any ) -> Union[Type, Tuple[Type]]:
        # https://stackoverflow.com/questions/45957615/check-a-variable-against-union-type-at-runtime-in-python-3-6
        if isinstance(input, Type):       return input
        if hasattr(input, '__origin__'):
            if input.__origin__ == Union: return tuple( cls._type(arg) for arg in input.__args__ )  # Union[int,str] -> (<class 'int'>, <class 'str'>)
            else:                         return input.__origin__  # Tuple[int,int] -> <class 'tuple'>
        if callable(input):               return cls.type( cls.inspect_signature(input).return_annotation )
        else:                             return type(input)


    @classmethod
    def isinstance( cls, a, b ):
        a_types = cls.types(a)
        b_types = cls.types(b)
        return any([ a_type == b_type for a_type, b_type in product(a_types, b_types) ])

    @staticmethod
    @lru_cache(None)
    def inspect_signature( input: Callable ):
        return inspect.signature(input)

    @classmethod
    def argument_permutations(cls, function: Callable, context: Context, arguments=[]):
        parameters  = cls.inspect_signature(function).parameters
        if len(parameters) == 0: return []
        parameters        = dict(parameters)
        arg_options       = defaultdict(list)
        arguments_by_type = cls.group_by_type(arguments)
        context_by_type   = cls.group_context_by_type(context)
        for index, (key, parameter) in enumerate(parameters.items()):
            index      = str(index)  # create_context() stores numbers as strings, and symbols() requires a string
            annotation = parameters[key].annotation
            for annotation_type in cls.types(annotation):

                # Type input as first parameter
                if index in context:
                    if cls.isinstance(context[index], annotation_type):
                        arg_options[key].append( Symbol(index) )
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
