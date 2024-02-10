"""Model Saving and Loading Tests.

"""

import pytest
import jax.numpy as jnp
import jax.random as jrandom
import os, shutil
import numpy as np

from tests.conftest import TMPDIR, DATDIR
from plnn.models import DeepPhiPLNN

#####################
##  Configuration  ##
#####################

def get_make_args(fpath):
    with open(fpath, 'r') as f:
        pair_list = [x[:-1].split("=") for x in f.readlines()]
        args = {k: eval(v) for k, v in pair_list}
    key = jrandom.PRNGKey(0)
    return args, key

###############################################################################
###############################   BEGIN TESTS   ###############################
###############################################################################

@pytest.mark.parametrize('arg_fpath', [
    f"{DATDIR}/make_args/make_args1.txt",
])
@pytest.mark.parametrize('dtype', [jnp.float32, jnp.float64])
@pytest.mark.parametrize('initialize', [True, False])
class TestMakeSaveLoad:
    
    def test_make(self, arg_fpath, dtype, initialize):
        args, key = get_make_args(arg_fpath)
        model, _ = DeepPhiPLNN.make_model(key, dtype=dtype, **args)
        if initialize: 
            model = model.initialize(key, dtype=dtype)
        errors = []
        if not isinstance(model, DeepPhiPLNN):
            msg = f"Wrong type for model Expected DeepPhiPLNN. Got {type(model)}."
            errors.append(msg)
        model_layers = model.get_linear_layer_parameters()
        if not np.all([l.dtype == dtype for l in model_layers]):
            msg = f"Wrong datatype for model layers. "
            msg += f"Got {[l.dtype for l in model_layers]}. Expected {dtype}"
            errors.append(msg)
        assert not errors, "Errors occurred:\n{}".format("\n".join(errors))

    def test_make_save(self, arg_fpath, dtype, initialize):
        args, key = get_make_args(arg_fpath)
        model, _ = DeepPhiPLNN.make_model(key, dtype=dtype, **args)
        if initialize: 
            model = model.initialize(key, dtype=dtype)
        tmpdir = f"{TMPDIR}/tmp_make_save"
        fpath = f"{tmpdir}/tmp_model.pth"
        os.makedirs(tmpdir, exist_ok=True)
        model.save(fpath, args)
        assert os.path.isfile(fpath)
        shutil.rmtree(tmpdir)
    
    def test_make_save_load(self, arg_fpath, dtype, initialize):
        args, key = get_make_args(arg_fpath)
        model, _ = DeepPhiPLNN.make_model(key, dtype=dtype, **args)
        if initialize: 
            model = model.initialize(key, dtype=dtype)
        tmpdir = f"{TMPDIR}/tmp_make_save_load"
        fpath = f"{tmpdir}/tmp_model.pth"
        os.makedirs(tmpdir, exist_ok=True)
        model.save(fpath, args)
        # Load another instance
        model2, hyperparams = DeepPhiPLNN.load(fpath, dtype=dtype)
        assert isinstance(model2, DeepPhiPLNN)
        shutil.rmtree(tmpdir)
