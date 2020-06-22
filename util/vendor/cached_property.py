# -*- coding: utf-8 -*-
# Source: https://github.com/pydanny/cached-property/blob/master/cached_property.py

# Simplified function for performance
class cached_property(object):
    """
    A property that is only computed once per instance and then replaces itself
    with an ordinary attribute. Deleting the attribute resets the property.
    Source: https://github.com/bottlepy/bottle/commit/fa7733e075da0d790d809aa3d2f53071897e6f76
    """

    def __init__(self, func):
        self.__doc__ = getattr(func, "__doc__")
        self.func = func
        self.func_name = self.func.__name__

    def __get__(self, obj, cls):
        if obj is None: return self
        value = obj.__dict__[self.func_name] = self.func(obj)
        return value
