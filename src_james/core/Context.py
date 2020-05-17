from collections import UserDict, UserList
from typing import Dict, Set

import numpy as np

from src_james.core.DataModel import Problem, Task



class Context(UserDict):
    excluded_keys  = { 'output', 'solution', 'raw', 'dtype', *dir(UserDict()), *dir(UserList()) }
    excluded_types = ( str,int,float,tuple,list,set,np.ndarray,type,Task )
    autoresolve    = ( 'input', 'task', 'filename', 'problemset', 'grids', 'outputs', 'test_or_train', 'inputs' )

    def __init__( self, problem: Problem, *args, **kwargs ):
        super().__init__(**kwargs)
        #if len(args) == 0 and 'input' in problem: args = [ problem['input'] ]
        arg_dict   = { str(index): arg for index,arg in enumerate(args) }
        self.data  = {
            **arg_dict,
            "problem":  problem,
            **kwargs,
        }
        self._version = 0
        for field in self.autoresolve:
            self.__getitem__(field)

        # assert not self.expand(recurse=True)    # auto-resolve (very slow)
        pass

    def expand( self ) -> Set[str]:
        start_keys = set(self.data.keys())
        for key, value in list(self.data.items()):
            if isinstance(value, self.excluded_types): continue
            if isinstance(value, (Dict,UserDict)):
                for name in value:
                    if name in self.excluded_keys: continue
                    if name in self.data:          continue
                    self.data[name] = value[name]

            attrs = set(dir(value)) - set(self.excluded_keys) - set(self.data.keys())
            for name in attrs:
                if name.startswith('_'):       continue
                self.data[name] = getattr(value, name)

        if set(self.data.keys()) - start_keys:
            self._version += 1
            self.expand()

        new_keys = set(self.data.keys()) - start_keys
        return new_keys

    def __lookup(self, key):
        if key in self.data: return self.data[key]

        for _, value in self.data.items():
            if hasattr(value, key):
                return getattr(value, key)
            if isinstance(value, (Dict,UserDict)) and key in value:
                return value[key]
        return None

    def __getitem__(self, key):
        if key not in self.data:
            value = self.__lookup(key)
            if value is not None: self.data[key] = value
        return self.data.get(key, None)

    def __contains__(self, key):
        return self.__getitem__(key) is not None

    def __str__(self):
        output = {}
        for key, value in self.data.items():
            if   isinstance(value, np.ndarray):        output[key] = np.array(value.shape)
            elif hasattr(self.data[key], '__dict__'):  output[key] = type(value)
            else:                                      output[key] = value
        return str(output)
