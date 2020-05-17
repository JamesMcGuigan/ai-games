import inspect
import traceback
from collections import UserDict, defaultdict
from itertools import product
from typing import Any, Callable, DefaultDict, Dict, List, Tuple, Type, Union

from src_james.core.Context import Context
from src_james.core.DataModel import Problem
from src_james.core.Symbol import Symbol
from src_james.settings import settings



class Rule(object):
    def __init__(self, function: Callable, arguments={}):
        self.function  = function
        self.arguments = arguments
        # self.context   = context


    def __call__(self, context: Union[Problem,Context]) -> Any:
        if not isinstance(context, Context): context = Context(context)
        output  = self.call_with_context(self.function, context, self.arguments)
        return output

    def __repr__(self):
        arguments = { key: value.__name__ if hasattr(value, '__name__') else str(value)
                      for key, value in self.arguments.items() }
        arguments = " ".join( f"{k}={v}" for k,v in arguments.items() )
        return f'<Rule {self.function.__name__}({arguments})>'


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
    def group_context_by_type( cls, context: Union[Dict,UserDict] ) -> DefaultDict[Type,List[Any]]:
        grouped = defaultdict(list)
        for name, item in context.items():
            types = cls.types(item)
            for type in types:
                grouped[type].append( Symbol(name) )
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
