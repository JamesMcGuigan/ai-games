import base64
import pickle
import tempfile

from numba import np
from pytest import fixture

from util.base64_file import *


@fixture
def data():
    return { "question": [ 0,2,1,0,0,0,0, 0,2,1,2,0,0,0 ], "answer": 42 }


def test_base64_file_varname():
    input    = './data/MontyCarloNode.pickle.zip.base64'
    expected = '_base64_file__MontyCarloNode__pickle__zip__base64'
    actual   = base64_file_varname(input)
    assert actual == expected


def test_base64_wrap_unwrap(data):
    varname     = base64_file_varname('test')
    input_bytes = base64.encodebytes(pickle.dumps(data))
    input_str   = base64_bytes_to_str(input_bytes)
    encoded     = base64_file_var_wrap(input_bytes, varname)
    decoded     = base64_file_var_unwrap(encoded)

    assert isinstance(input_bytes, bytes)
    assert isinstance(input_str,   str)
    assert isinstance(encoded, str)
    assert isinstance(decoded, str)
    assert varname     in encoded
    assert varname not in decoded
    assert input_str != encoded
    assert input_str == decoded


def test_base64_save_load(data):
    filename = '/tmp/test_base64_save_load'
    if os.path.exists(filename): os.remove(filename)
    assert not os.path.exists(filename)
    assert data == data

    filesize = base64_file_save(data, filename)
    output   = base64_file_load(filename)

    assert os.path.exists(filename)
    assert filesize < 1/1024  # less than 1kb
    assert data == output

