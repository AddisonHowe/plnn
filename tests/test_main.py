"""Main Entrypoint Tests

"""

import pytest
import shutil
import jax.numpy as jnp
import numpy as np
from contextlib import nullcontext as does_not_raise

from tests.conftest import DATDIR, TMPDIR, remove_dir

from plnn.main import parse_args, main
from plnn.models import DeepPhiPLNN, VAEPLNN
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

# @pytest.mark.skip()
@pytest.mark.parametrize('argstring_fpath', [
    f"{DATDIR}/test_main_args/argstring1.txt",
    f"{DATDIR}/test_main_args/argstring2.txt",
    f"{DATDIR}/test_main_args/argstring3.txt",
    f"{DATDIR}/test_main_args/argstring4.txt",
    f"{DATDIR}/test_main_args/argstring_algphi1.txt",
    f"{DATDIR}/test_main_args/argstring_vaeplnn1.txt",
])
def test_main(argstring_fpath):
    argstring = get_args(argstring_fpath)
    args = parse_args(argstring)
    args.outdir = f"{TMPDIR}/{args.outdir}"
    args.training_data = f"{DATDIR}/{args.training_data}"
    args.validation_data = f"{DATDIR}/{args.validation_data}"
    main(args)
    remove_dir(args.outdir)

# @pytest.mark.skip()
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

# @pytest.mark.skip()
@pytest.mark.parametrize('argstring_fpath, modelname, modelclass, dtype', [
    [f"{DATDIR}/test_main_args/argstring1.txt", "model1", DeepPhiPLNN, jnp.float32],
    [f"{DATDIR}/test_main_args/argstring2.txt", "model2", DeepPhiPLNN, jnp.float32],
    [f"{DATDIR}/test_main_args/argstring3.txt", "model3", DeepPhiPLNN, jnp.float32],
    [f"{DATDIR}/test_main_args/argstring4.txt", "model4", DeepPhiPLNN, jnp.float32],
    [f"{DATDIR}/test_main_args/argstring_vaeplnn1.txt", "modelvae1", VAEPLNN, jnp.float32],
])
def test_reproducibility(argstring_fpath, modelname, modelclass, dtype):
    argstring = get_args(argstring_fpath)
    args = parse_args(argstring)
    args.outdir = f"{TMPDIR}/{args.outdir}"
    args.training_data = f"{DATDIR}/{args.training_data}"
    args.validation_data = f"{DATDIR}/{args.validation_data}"
    assert args.seed > 0, f"This test requires a seed! See {argstring_fpath}."
    main(args)
    # Load first model instance
    model1, hp1 = modelclass.load(
        f"{args.outdir}/states/{modelname}_1.pth",
        dtype=dtype
    )
    remove_dir(f"{TMPDIR}/tmp_test_main")
    # Re-run main
    main(args)
    model2, hp2 = modelclass.load(
        f"{args.outdir}/states/{modelname}_1.pth",
        dtype=dtype
    )
    remove_dir(f"{TMPDIR}/tmp_test_main")

    w1 = model1.get_parameters()['phi.w']
    w2 = model2.get_parameters()['phi.w']
    for arr1, arr2 in zip(w1, w2):
        assert np.allclose(arr1, arr2), "Mismatch between trained parameters"

# @pytest.mark.skip()
@pytest.mark.parametrize('argstring_fpath, expected_hist', [
    [f"{DATDIR}/test_main_args/argstring_dt_schedule1.txt",
     [0.5, 0.25, 0.25, 0.5, 0.25]],
])
def test_dt_schedule(argstring_fpath, expected_hist):
    argstring = get_args(argstring_fpath)
    args = parse_args(argstring)
    args.outdir = f"{TMPDIR}/{args.outdir}"
    args.training_data = f"{DATDIR}/{args.training_data}"
    args.validation_data = f"{DATDIR}/{args.validation_data}"
    main(args)
    dt_hist = np.load(f"{args.outdir}/dt_hist.npy")
    remove_dir(args.outdir)
    assert np.all(dt_hist == expected_hist), \
        f"Expected {expected_hist}.\nGot {dt_hist}."

# @pytest.mark.skip()
@pytest.mark.parametrize('argstring_fpath, expected_hist', [
    # 5 epochs, lr 0.1->0.01, warmup: 1, decay: -1, batch_size: 100/100
    [f"{DATDIR}/test_main_args/argstring_lr_schedule1.txt",
     [0.1, 0.1, 0.056234132519, 0.0316227766017, 0.0177827941004]],

    # 5 epochs, lr 0.1->0.01, warmup: 1, decay: -1, batch_size: 200 > 100
    [f"{DATDIR}/test_main_args/argstring_lr_schedule2.txt",
     [0.1, 0.1, 0.056234132519, 0.0316227766017, 0.0177827941004]],

    # 5 epochs, lr 0.1->0.01, warmup: 1, decay: -1, batch_size: 50 < 100
    [f"{DATDIR}/test_main_args/argstring_lr_schedule3.txt",
     [0.1, 0.0749894209332, 0.0421696503429, 0.0237137370566, 0.0133352143216]],

    # 5 epochs, lr 0.1->0.01, warmup: 1, decay: 4, batch_size: 100/100
     [f"{DATDIR}/test_main_args/argstring_lr_schedule4.txt",
     [0.1, 0.1, 0.056234132519, 0.0316227766017, 0.0177827941004]],

    # 5 epochs, lr 0.1->0.01, warmup: 1, decay: 10, batch_size: 100/100
    [f"{DATDIR}/test_main_args/argstring_lr_schedule5.txt",
    [0.1, 0.1, 0.0794328234724, 0.063095734448, 0.0501187233627]],

    # 5 epochs, lr 0.1->0.01, warmup: 1, decay: -1, batch_size: 20 < 100
    [f"{DATDIR}/test_main_args/argstring_lr_schedule6.txt",
     [0.1, 0.063095734448, 0.0354813389234, 0.0199526231497, 0.011220184543]],

    # 5 epochs, lr 0.1->0.01, warmup: 3, decay: -1, batch_size: 200 > 100
    [f"{DATDIR}/test_main_args/argstring_lr_schedule7.txt",
     [0.1, 0.1, 0.1, 0.1, 0.0316227766017]],
])
def test_lr_schedule(argstring_fpath, expected_hist):
    argstring = get_args(argstring_fpath)
    args = parse_args(argstring)
    args.outdir = f"{TMPDIR}/{args.outdir}"
    args.training_data = f"{DATDIR}/{args.training_data}"
    args.validation_data = f"{DATDIR}/{args.validation_data}"
    main(args)
    lr_hist = np.load(f"{args.outdir}/learning_rate_history.npy")
    remove_dir(args.outdir)
    assert np.allclose(lr_hist, expected_hist), \
        f"Expected {expected_hist}.\nGot {lr_hist}."
