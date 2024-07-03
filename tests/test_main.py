"""Main Entrypoint Tests

"""

import pytest
import shutil
import jax.numpy as jnp
import numpy as np
from contextlib import nullcontext as does_not_raise

from tests.conftest import DATDIR, TMPDIR, remove_dir

from plnn.main import parse_args, main
from plnn.models import DeepPhiPLNN
from plnn.dataset import NumpyLoader
from plnn.dataset import LandscapeSimulationDataset as Dataset

#####################
##  Configuration  ##
#####################

def get_args(fpath):
    with open(fpath, 'r') as f:
        argstring = f.readline()
        arglist = argstring.split(" ")
        return arglist
        
###############################################################################
###############################   BEGIN TESTS   ###############################
###############################################################################

@pytest.mark.parametrize('argstring_fpath', [
    f"{DATDIR}/test_main_args/argstring1.txt",
    f"{DATDIR}/test_main_args/argstring2.txt",
    f"{DATDIR}/test_main_args/argstring3.txt",
    f"{DATDIR}/test_main_args/argstring4.txt",
    f"{DATDIR}/test_main_args/argstring_algphi1.txt",
])
def test_main(argstring_fpath):
    argstring = get_args(argstring_fpath)
    args = parse_args(argstring)
    args.outdir = f"{TMPDIR}/{args.outdir}"
    args.training_data = f"{DATDIR}/{args.training_data}"
    args.validation_data = f"{DATDIR}/{args.validation_data}"
    main(args)
    remove_dir(args.outdir)


@pytest.mark.parametrize(
        "argstring_fpath, expect_warning_context, expect_error_context", [
    # Dataset does not sample. Use 10 cells internal. Model samples.
    [f"{DATDIR}/test_main_args/argstring_nh_1.txt", 
     pytest.warns(RuntimeWarning, match=Dataset.nonhomogeneous_warning+"*"),
     pytest.raises(RuntimeError, match=NumpyLoader.nonloadable_error_message),
     ],
    # Dataset samples 10 cells. Use 10 cells internal. Model samples.
    [f"{DATDIR}/test_main_args/argstring_nh_2.txt", 
     does_not_raise(), 
     does_not_raise()],
    # Dataset samples 10 cells. Use 10 cells internal. Model does not sample.
    [f"{DATDIR}/test_main_args/argstring_nh_3.txt", 
     does_not_raise(), 
     does_not_raise()],
])
def test_main_nonhomogeneous(
    argstring_fpath, expect_warning_context, expect_error_context
):
    with expect_warning_context, expect_error_context:
            argstring = get_args(argstring_fpath)
            args = parse_args(argstring)
            args.outdir = f"{TMPDIR}/{args.outdir}"
            args.training_data = f"{DATDIR}/{args.training_data}"
            args.validation_data = f"{DATDIR}/{args.validation_data}"
            main(args)
    remove_dir(args.outdir)


@pytest.mark.parametrize('argstring_fpath, modelname, dtype', [
    [f"{DATDIR}/test_main_args/argstring1.txt", "model1", jnp.float32],
    [f"{DATDIR}/test_main_args/argstring2.txt", "model2", jnp.float32],
    [f"{DATDIR}/test_main_args/argstring3.txt", "model3", jnp.float32],
    [f"{DATDIR}/test_main_args/argstring4.txt", "model4", jnp.float32],
])
def test_reproducibility(argstring_fpath, modelname, dtype):
    argstring = get_args(argstring_fpath)
    args = parse_args(argstring)
    args.outdir = f"{TMPDIR}/{args.outdir}"
    args.training_data = f"{DATDIR}/{args.training_data}"
    args.validation_data = f"{DATDIR}/{args.validation_data}"
    assert args.seed > 0, f"This test requires a seed! See {argstring_fpath}."
    main(args)
    # Load first model instance
    model1, hp1 = DeepPhiPLNN.load(
        f"{args.outdir}/states/{modelname}_1.pth",
        dtype=dtype
    )
    remove_dir(f"{TMPDIR}/tmp_test_main")
    # Re-run main
    main(args)
    model2, hp2 = DeepPhiPLNN.load(
        f"{args.outdir}/states/{modelname}_1.pth",
        dtype=dtype
    )
    remove_dir(f"{TMPDIR}/tmp_test_main")

    w1 = model1.get_parameters()['phi.w']
    w2 = model2.get_parameters()['phi.w']
    for arr1, arr2 in zip(w1, w2):
        assert np.allclose(arr1, arr2), "Mismatch between trained parameters"
