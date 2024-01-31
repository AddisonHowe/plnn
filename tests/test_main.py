import pytest
import os, glob, shutil
import numpy as np
from plnn.main import parse_args, main


def get_args(fpath):
    with open(fpath, 'r') as f:
        argstring = f.readline()
        arglist = argstring.split(" ")
        return arglist
    
def _remove_dir(dir):
    shutil.rmtree(dir)
    

@pytest.mark.parametrize('argstring_fpath', [
    "tests/test_main_args/argstring1.txt",
    "tests/test_main_args/argstring2.txt",
    "tests/test_main_args/argstring3.txt",
    "tests/test_main_args/argstring4.txt",
])
def test_main(argstring_fpath):
    argstring = get_args(argstring_fpath)
    args = parse_args(argstring)
    main(args)
    _remove_dir("tests/tmp_test_main")
