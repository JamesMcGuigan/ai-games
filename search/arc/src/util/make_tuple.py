
def make_tuple(args):
    if isinstance(args, tuple): return args
    if isinstance(args, list):  return tuple(args)
    return (args,)