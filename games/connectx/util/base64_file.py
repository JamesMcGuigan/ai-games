import base64
import gzip
import os
import re
import time
from typing import Any
from typing import Union

import dill
import humanize


# _base64_file__test_base64_static_import = """
# H4sIAPx9LF8C/2tgri1k0IjgYGBgKCxNLS7JzM8rZIwtZNLwZvBm8mYEkjAI4jFB2KkRbED1iXnF
# 5alFhczeWqV6AEGfwmBHAAAA
# """


def base64_file_varname(filename: str) -> str:
    # ../data/AntColonyTreeSearchNode.dill.zip.base64 -> _base64_file__AntColonyTreeSearchNode__dill__zip__base64
    varname = re.sub(r'^.*/',   '',   filename)  # remove directories
    varname = re.sub(r'[.\W]+', '__', varname)   # convert dots and non-ascii to __
    varname = f"_base64_file__{varname}"
    return varname


def base64_file_var_wrap(base64_data: Union[str,bytes], varname: str) -> str:
    return f'{varname} = """\n{base64_data.strip()}\n"""'                    # add varname = """\n\n""" wrapper


def base64_file_var_unwrap(base64_data: str) -> str:
    output = base64_data.strip()
    output = re.sub(r'^\w+ = """|"""$', '', output)  # remove varname = """ """ wrapper
    output = output.strip()
    return output


def base64_file_encode(data: Any) -> str:
    encoded = dill.dumps(data)
    encoded = gzip.compress(encoded)
    encoded = base64.encodebytes(encoded).decode('utf8').strip()
    return encoded


def base64_file_decode(encoded: str) -> Any:
    data = base64.b64decode(encoded)
    data = gzip.decompress(data)
    data = dill.loads(data)
    return data


def base64_file_save(data: Any, filename: str, vebose=True) -> float:
    """
        Saves a base64 encoded version of data into filename, with a varname wrapper for importing via kaggle_compile.py
        # Doesn't create/update global variable.
        Returns filesize in bytes
    """
    varname    = base64_file_varname(filename)
    start_time = time.perf_counter()
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as file:
            encoded = base64_file_encode(data)
            output  = base64_file_var_wrap(encoded, varname)
            output  = output.encode('utf8')
            file.write(output)
            file.close()
        if varname in globals(): globals()[varname] = encoded  # globals not shared between modules, but update for saftey

        filesize = os.path.getsize(filename)
        if vebose:
            time_taken = time.perf_counter() - start_time
            print(f"base64_file_save(): {filename:40s} | {humanize.naturalsize(filesize)} in {time_taken:4.1f}s")
        return filesize
    except Exception as exception:
        pass
    return 0.0


def base64_file_load(filename: str, vebose=True) -> Union[Any,None]:
    """
        Performs a lookup to see if the global variable for this file alread exists
        If not, reads the base64 encoded file from filename, with an optional varname wrapper
        # Doesn't create/update global variable.
        Returns decoded data
    """
    varname    = base64_file_varname(filename)
    start_time = time.perf_counter()
    try:
        # Hard-coding PyTorch weights into a script - https://www.kaggle.com/c/connectx/discussion/126678
        encoded = None

        if varname in globals():
            encoded = globals()[varname]

        if encoded is None and os.path.exists(filename):
            with open(filename, 'rb') as file:
                encoded = file.read().decode('utf8')
                encoded = base64_file_var_unwrap(encoded)
                # globals()[varname] = encoded  # globals are not shared between modules

        if encoded is not None:
            data = base64_file_decode(encoded)

            if vebose:
                filesize = os.path.getsize(filename)
                time_taken = time.perf_counter() - start_time
                print(f"base64_file_load(): {filename:40s} | {humanize.naturalsize(filesize)} in {time_taken:4.1f}s")
            return data
    except Exception as exception:
        print(f'base64_file_load({filename}): Exception:', exception)
    return None
