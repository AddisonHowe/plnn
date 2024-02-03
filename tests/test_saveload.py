import pytest
import jax.numpy as jnp
import jax.random as jrandom
import os, glob, shutil
import numpy as np
from plnn.models import PLNN, make_model, initialize_model, save_model, load_model


def get_make_args(fpath):
    with open(fpath, 'r') as f:
        pair_list = [x[:-1].split("=") for x in f.readlines()]
        args = {k: eval(v) for k, v in pair_list}
    key = jrandom.PRNGKey(0)
    return args, key

@pytest.mark.parametrize('arg_fpath', [
    "tests/make_args/make_args1.txt",
])
@pytest.mark.parametrize('dtype', [jnp.float32, jnp.float64])
@pytest.mark.parametrize('initialize', [True, False])
class TestMakeSaveLoad:
    
    def test_make(self, arg_fpath, dtype, initialize):
        args, key = get_make_args(arg_fpath)
        model, _ = make_model(key, dtype=dtype, **args)
        if initialize: 
            model = initialize_model(key, model, dtype=dtype)
        errors = []
        if not isinstance(model, PLNN):
            msg = f"Wrong type for model Expected PLNN. Got {type(model)}."
            errors.append(msg)
        model_layers = model.get_linear_layer_parameters()
        if not np.all([l.dtype == dtype for l in model_layers]):
            msg = f"Wrong datatype for model layers. "
            msg += f"Got {[l.dtype for l in model_layers]}. Expected {dtype}"
            errors.append(msg)
        assert not errors, "Errors occurred:\n{}".format("\n".join(errors))

    def test_make_save(self, arg_fpath, dtype, initialize):
        args, key = get_make_args(arg_fpath)
        model, _ = make_model(key, dtype=dtype, **args)
        if initialize: 
            model = initialize_model(key, model, dtype=dtype)
        tmpdir = "tests/tmp_make_save"
        fpath = f"{tmpdir}/tmp_model.pth"
        os.makedirs(tmpdir, exist_ok=True)
        save_model(fpath, model, args)
        assert os.path.isfile(fpath)
        shutil.rmtree(tmpdir)
    
    def test_make_save_load(self, arg_fpath, dtype, initialize):
        args, key = get_make_args(arg_fpath)
        model, _ = make_model(key, dtype=dtype, **args)
        if initialize: 
            model = initialize_model(key, model, dtype=dtype)
        tmpdir = "tests/tmp_make_save_load"
        fpath = f"{tmpdir}/tmp_model.pth"
        os.makedirs(tmpdir, exist_ok=True)
        save_model(fpath, model, args)
        # Load another instance
        model2, hyperparams = load_model(fpath, dtype=dtype)
        assert isinstance(model2, PLNN)
        shutil.rmtree(tmpdir)
