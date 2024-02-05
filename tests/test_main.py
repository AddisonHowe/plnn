import pytest
import shutil
import jax.numpy as jnp
import numpy as np
from plnn.main import parse_args, main
from plnn.models import DeepPhiPLNN


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


@pytest.mark.parametrize('argstring_fpath, modelname, dtype', [
    ["tests/test_main_args/argstring1.txt", "model1", jnp.float32],
    ["tests/test_main_args/argstring2.txt", "model2", jnp.float32],
    ["tests/test_main_args/argstring3.txt", "model3", jnp.float32],
    ["tests/test_main_args/argstring4.txt", "model4", jnp.float32],
])
def test_reproducibility(argstring_fpath, modelname, dtype):
    argstring = get_args(argstring_fpath)
    args = parse_args(argstring)
    assert args.seed > 0, f"This test requires a seed! See {argstring_fpath}."
    main(args)
    # Load first model instance
    model1, hp1 = DeepPhiPLNN.load(
        f"tests/tmp_test_main/{modelname}/states/{modelname}_1.pth",
        dtype=dtype
    )
    _remove_dir("tests/tmp_test_main")
    # Re-run main
    main(args)
    model2, hp2 = DeepPhiPLNN.load(
        f"tests/tmp_test_main/{modelname}/states/{modelname}_1.pth",
        dtype=dtype
    )
    _remove_dir("tests/tmp_test_main")

    w1 = model1.get_parameters()['phi.w']
    w2 = model2.get_parameters()['phi.w']
    for arr1, arr2 in zip(w1, w2):
        assert np.allclose(arr1, arr2), "Mismatch between trained parameters"
