import pytest
import jax.numpy as jnp
import jax.random as jrandom
import os, glob, shutil
import numpy as np
from plnn.models import PLNN, make_model, save_model, load_model


def get_make_args(fpath):
    with open(fpath, 'r') as f:
        pair_list = [x[:-1].split("=") for x in f.readlines()]
        args = {k: eval(v) for k, v in pair_list}
    key = jrandom.PRNGKey(0)
    return args, key

@pytest.mark.parametrize('arg_fpath', [
    "tests/make_args/make_args1.txt",
])
class TestMakeSaveLoad:
    
    def test_make(self, arg_fpath):
        args, key = get_make_args(arg_fpath)
        model, _ = make_model(key, dtype=jnp.float32, **args)
        assert isinstance(model, PLNN)

    def test_make_save(self, arg_fpath):
        args, key = get_make_args(arg_fpath)
        model, _ = make_model(key, **args)
        tmpdir = "tests/tmp_make_save"
        fpath = f"{tmpdir}/tmp_model.pth"
        os.makedirs(tmpdir)
        save_model(fpath, model, args)
        assert os.path.isfile(fpath)
        shutil.rmtree(tmpdir)
    
    def test_make_save_load(self, arg_fpath):
        args, key = get_make_args(arg_fpath)
        model, _ = make_model(key, **args)
        tmpdir = "tests/tmp_make_save_load"
        fpath = f"{tmpdir}/tmp_model.pth"
        os.makedirs(tmpdir)
        save_model(fpath, model, args)
        # Load another instance
        model2, hyperparams = load_model(fpath)
        assert isinstance(model2, PLNN)
        shutil.rmtree(tmpdir)
