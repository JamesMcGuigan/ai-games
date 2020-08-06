import base64
import gzip
import os
import pickle
import re
import time
from typing import Any, Union


def get_filesize_mb(filename: str) -> float:
    return os.path.getsize(filename)/1024/1024


def base64_bytes_to_str(bytes: Union[bytes,str]) -> str:
    return re.sub(r"^b'(.*)(\\n)*'$", r'\1', str(bytes), re.DOTALL)  # remove b'' wrapper and encoded trailing \\n


def base64_file_varname(filename: str) -> str:
    # ../data/AntColonyTreeSearchNode.pickle.zip.base64 -> _base64_file__AntColonyTreeSearchNode__pickle__zip__base64
    varname = re.sub(r'^.*/',     '',   filename)  # remove directories
    varname = re.sub(r'[\.\W]+',  '__', varname)   # convert dots and non-ascii to __
    varname = f"_base64_file__{varname}"
    return varname


def base64_file_var_wrap(base64_data: Union[str,bytes], varname: str) -> str:
    return f'{varname} = """\n{base64_data}\n"""'                    # add varname = """\n\n""" wrapper


def base64_file_var_unwrap(base64_data: str) -> str:
    output = re.sub(r'^\n*\w+ = """\n*(.*)\n*"""\n*$', r'\1', base64_data, re.DOTALL)  # remove varname = """\n\n""" wrapper
    output = base64_bytes_to_str(output)
    return output


def base64_file_load(filename: str, vebose=True) -> Union[Any,None]:
    varname    = base64_file_varname(filename)
    start_time = time.perf_counter()
    try:
        # Hard-coding PyTorch weights into a script - https://www.kaggle.com/c/connectx/discussion/126678
        data = None
        if varname in globals():
            data = globals()[varname]
        if data is None and os.path.exists(filename):
            with open(filename, 'rb') as file:
                data = file.read()
        if data is not None:
            data = base64.b64decode(data)
            data = gzip.decompress(data)
            data = pickle.loads(data)

            if vebose:
                time_taken = time.perf_counter() - start_time
                print(f"base64_file_load(): {filename:40s} | {get_filesize_mb(filename):4.1f}MB in {time_taken:4.1f}s")
            return data
    except Exception as exception:
        print(f'base64_file_load({filename}): Exception:', exception)
    return None


def base64_file_save(data: Any, filename: str, vebose=True) -> float:
    """ Returns size of file in Mb """
    varname    = base64_file_varname(filename)
    start_time = time.perf_counter()
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as file:
            data = pickle.dumps(data)
            data = gzip.compress(data)
            data = base64.encodebytes(data)
            file.write(data)
            file.close()
        filesize = get_filesize_mb(filename)
        if vebose:
            time_taken = time.perf_counter() - start_time
            print(f"base64_file_save(): {filename:40s} | {filesize:4.1f}MB in {time_taken:4.1f}s")
        return filesize
    except Exception as exception:
        pass
    return 0.0