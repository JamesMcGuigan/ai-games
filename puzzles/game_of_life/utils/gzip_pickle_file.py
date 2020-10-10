import gzip
import os
import pickle
from typing import Any

import humanize


def read_gzip_pickle_file(filename: str) -> Any:
    try:
        if not os.path.exists(filename): raise FileNotFoundError
        with open(filename, 'rb') as file:
            data = file.read()
            try:    data = gzip.decompress(data)
            except: pass
            data = pickle.loads(data)
    except Exception as exception:
        data = None
    return data


def save_gzip_pickle_file(data: Any, filename: str, verbose=True) -> int:
    try:
        with open(filename, 'wb') as file:
            data = pickle.dumps(data)
            data = gzip.compress(data)
            file.write(data)
            file.close()
        filesize = os.path.getsize(filename)
        if verbose: print(f'wrote: {filename} = {humanize.naturalsize(filesize)}')
        return filesize
    except:
        return 0
