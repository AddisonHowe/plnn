"""Tests of Data Generation Modules.

"""

import pytest
import shutil
import numpy as np

from tests.conftest import DATDIR, TMPDIR, remove_dir
from plnn.data_generation.generate_data import parse_args, main

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
    f"{DATDIR}/test_data_generation/argstring1.txt",
])
@pytest.mark.filterwarnings(
    "ignore:divide by zero encountered in scalar divide",
    "ignore:invalid value encountered in multiply"
)
def test_data_generation(argstring_fpath):
    argstring = get_args(argstring_fpath)
    args = parse_args(argstring)
    args.outdir = f"{TMPDIR}/{args.outdir}"
    main(args)
    outdir = args.outdir
    dt_save = args.dt_save
    ani_dt = args.animation_dt
    xs_save = np.load(f"{outdir}/sim0/xs.npy")
    xs_anim = np.load(f"{outdir}/sim0/ani_xs.npy")
    ts_save = np.load(f"{outdir}/sim0/ts.npy")
    ts_anim = np.load(f"{outdir}/sim0/ani_ts.npy")
    assert xs_save.shape == (len(ts_save), args.ncells, 2)
    errors = []
    for i in range(len(xs_save)):
        j = i * int(dt_save // ani_dt)
        tsim = ts_save[i]
        tani = ts_anim[j]
        xsim = xs_save[i]
        xani = xs_anim[j]
        if not np.allclose(xsim, xani):
            msg = f"Mismatch between x values at tsim={tsim}, tani={tani}."
            errors.append(msg)
    assert not errors, "Errors occurred:\n{}".format("\n".join(errors))    
    remove_dir(f"{TMPDIR}/tmp_test_data_generation")
