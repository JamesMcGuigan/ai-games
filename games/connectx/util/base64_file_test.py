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
    varname   = base64_file_varname('test')
    input     = base64.encodebytes(pickle.dumps(data)).decode('utf8').strip()
    wrapped   = base64_file_var_wrap(input, varname)
    unwrapped = base64_file_var_unwrap(wrapped)

    assert isinstance(input,   str)
    assert isinstance(wrapped, str)
    assert isinstance(unwrapped, str)
    assert varname     in wrapped
    assert varname not in unwrapped
    assert input != wrapped
    assert input == unwrapped


def test_base64_save_load(data):
    assert data == data

    filename = '/tmp/test_base64_save_load'
    if os.path.exists(filename): os.remove(filename)
    assert not os.path.exists(filename)

    loaded   = base64_file_load(filename)
    assert loaded is None

    varname  = base64_file_varname(filename)
    filesize = base64_file_save(data, filename)
    loaded   = base64_file_load(filename)

    assert os.path.exists(filename)
    assert filesize < 1024  # less than 1kb
    assert data == loaded

    # assert varname in globals()          # globals are not shared between modules
    # assert output == globals()[varname]  # globals are not shared between modules


def test_base64_static_import(data):
    assert data == data

    filename = '/tmp/test_base64_static_import'
    if os.path.exists(filename): os.remove(filename)
    assert not os.path.exists(filename)


    varname  = base64_file_varname(filename)
    filesize = base64_file_save(data, filename)
    loaded   = base64_file_load(filename)

    if varname in globals(): del globals()[varname]  # globals are not shared between modules
    contents = open(filename, 'r').read()
    exec(contents, globals())
    assert varname in globals()
    encoded = globals()[varname]
    decoded = base64_file_decode(encoded)

    assert varname in contents
    assert data == loaded == decoded




