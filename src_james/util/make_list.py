from collections import Iterable


def make_list(args):
    if isinstance(args, (list,Iterable)):
        return list(args)
    else:
        return list(args)