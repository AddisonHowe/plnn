"""Tests for custom Dataloader and Dataset.

Uses synthetic datasets located in tests/_data/test_dataloader/
    dataset1: 8 simulations, each with 20 2-dimensional cells at 11 timepoints
        between 0 and 10. Cells are ordered lexicographically by simulations 
        and timepoints, with coords (0, 1), (1, 2), ... etc.
    dataset2: 8 simulations, with [10, 20, 10, 20, 10, 20 10, 20] cells. Same
        timepoints as dataset1.
"""

import pytest
import numpy as np
from contextlib import nullcontext as does_not_raise

from tests.conftest import DATDIR

from plnn.dataset import LandscapeSimulationDataset, NumpyLoader

#####################
##  Configuration  ##
#####################

datdir = f"{DATDIR}/test_dataloader"

###############################################################################
###############################   BEGIN TESTS   ###############################
###############################################################################


@pytest.mark.parametrize("datdir, nsims, ntimes, ncells, ndims, sigparamshape", [
    [f"{datdir}/dataset1", 8, 11, 20, 2, (2,4)],
])
class TestHomogeneousDataset:

    def _load_dataset(self, datdir, nsims, ndims, ncells_sample):
        self.dataset = LandscapeSimulationDataset(
            datdir, 
            nsims=nsims,
            ndims=ndims,
            transform=None,
            ncells_sample=ncells_sample,
            seed=1,
        )
        return self.dataset
    
    def _check_length(self, length_expected):
        if len(self.dataset) != length_expected:
            return f"Expected len {length_expected}. Got {len(self.dataset)}."
    
    def _check_item_shapes(self, shape_t_exp, shape_x_exp, shape_sigparams_exp):
        errors = set()
        for item in self.dataset:
            if len(item) != 2:
                msg = f"Encountered item in dataset not of length 2."
                errors.add(msg)
            else:
                inputs, outputs = item
                # Check inputs should have 4 values with following shapes...
                if len(inputs) != 4:
                    msg = f"Encountered inputs in dataset not of length 4."
                    errors.add(msg)
                t0, x0, t1, sigparams = inputs
                x1 = outputs
                if t0.shape != shape_t_exp:
                    errors.add(f"Encountered incorrect shape of t0.")
                if t1.shape != shape_t_exp:
                    errors.add(f"Encountered incorrect shape of t1.")
                if x0.shape != shape_x_exp:
                    errors.add(f"Encountered incorrect shape of x0.")
                if x1.shape != shape_x_exp:
                    errors.add(f"Encountered incorrect shape of x1.")
                if sigparams.shape != shape_sigparams_exp:
                    errors.add(f"Encountered incorrect shape of sigparams.")
        return list(errors)
    
    def _check_xs_values(self, ncells, ndims, ntimes):
        errors = set()
        for i, item in enumerate(self.dataset):
            x0_exp = np.zeros([ncells, ndims])
            x1_exp = np.zeros([ncells, ndims])
            x0_exp[:,0] = np.arange(i*ncells, (i+1)*ncells)
            x0_exp[:,0] += ncells * (i // (ntimes - 1))  # skips at sim end.
            x0_exp[:,1] = 1 + x0_exp[:,0]  
            x1_exp[:,0] = np.arange((i+1)*ncells, (i+2)*ncells)
            x1_exp[:,0] += ncells * (i // (ntimes - 1))
            x1_exp[:,1] = 1 + x1_exp[:,0]

            inputs, x1 = item
            x0 = inputs[1]
            if not np.allclose(x0, x0_exp):
                errors.add("Encountered incorrect values in x0")
            if not np.allclose(x1, x1_exp):
                errors.add("Encountered incorrect values in x1")
        return list(errors)
    
    def _check_xs_types(self, ncells, ndims, ntimes, dtype_exp):
        errors = set()
        for i, item in enumerate(self.dataset):
            inputs, x1 = item
            x0 = inputs[1]
            if not isinstance(x0, np.ndarray):
                errors.add("x0 is not a ndarray")
            elif not isinstance(x0.flatten()[0], dtype_exp):
                msg = f"Encountered incorrect type for x0 data. "
                msg += f"Expected {dtype_exp}. Got {type(x0.flatten()[0])}."
                errors.add(msg)
            if not isinstance(x1, np.ndarray):
                errors.add("x0 is not a ndarray")
            elif not isinstance(x1.flatten()[0], dtype_exp):
                msg = f"Encountered incorrect type for x1 data. "
                msg += f"Expected {dtype_exp}. Got {type(x1.flatten()[0])}."
                errors.add(msg)
        return list(errors)

    def test_length_and_shapes(
            self, datdir, nsims, ntimes, ncells, ndims, sigparamshape
    ):
        self._load_dataset(datdir, nsims, ndims, 0)
        errors = []
        errmsg = self._check_length(nsims * (ntimes - 1))
        if errmsg:
            errors.append(errmsg)
        shape_errors = self._check_item_shapes(
            (), (ncells, ndims), sigparamshape
        )
        errors += shape_errors
        assert not errors, "Errors occurred:\n{}".format("\n".join(errors))

    def test_values(
            self, datdir, nsims, ntimes, ncells, ndims, sigparamshape
    ):
        self._load_dataset(datdir, nsims, ndims, 0)
        errors = []
        xs_value_errors = self._check_xs_values(ncells, ndims, ntimes)
        errors += xs_value_errors
        assert not errors, "Errors occurred:\n{}".format("\n".join(errors))

    def test_types(self, datdir, nsims, ntimes, ncells, ndims, sigparamshape):
        self._load_dataset(datdir, nsims, ndims, 0)
        errors = []
        xs_type_errors = self._check_xs_types(ncells, ndims, ntimes, np.float64)
        errors += xs_type_errors
        assert not errors, "Errors occurred:\n{}".format("\n".join(errors))


@pytest.mark.parametrize(
    "datdir, nsims, ntimes, ndims, sigparamshape", [
    [f"{datdir}/dataset2", 8, 11, 2, (2, 4)],
])
class TestNonhomogeneousDataset:

    def _load_dataset(self, datdir, nsims, ndims, transform, ncells_sample,
                      suppress_warnings=False,):
        self.dataset = LandscapeSimulationDataset(
            datdir, 
            nsims=nsims,
            ndims=ndims,
            transform=transform,
            ncells_sample=ncells_sample,
            seed=1,
            suppress_warnings=suppress_warnings,
        )
        return self.dataset
    
    def _check_length(self, length_expected):
        if len(self.dataset) != length_expected:
            return f"Expected len {length_expected}. Got {len(self.dataset)}."
    
    def _check_item_shapes(
            self, shape_t_exp, shape_sigparams_exp, transform, ncells_sample,
    ):
        errors = set()
        expected_x_shape = None
        if transform == 'sample' and isinstance(ncells_sample, int):
            expected_x_shape = (ncells_sample, 2)
        for item in self.dataset:
            if len(item) != 2:
                msg = f"Encountered item in dataset not of length 2."
                errors.append(msg)
            else:
                inputs, x1 = item
                # Check inputs should have 4 values with following shapes...
                if len(inputs) != 4:
                    msg = f"Encountered inputs in dataset not of length 4."
                    errors.append(msg)
                t0, x0, t1, sigparams = inputs
                if t0.shape != shape_t_exp:
                    errors.append(f"Encountered incorrect shape of t0.")
                if t1.shape != shape_t_exp:
                    errors.append(f"Encountered incorrect shape of t1.")
                if sigparams.shape != shape_sigparams_exp:
                    errors.append(f"Encountered incorrect shape of sigparams.")
                if expected_x_shape and expected_x_shape != x0.shape:
                    errors.append(f"Encountered incorrect shape of sampled x0.")
                if expected_x_shape and expected_x_shape != x1.shape:
                    errors.append(f"Encountered incorrect shape of sampled x1.")
        return list(errors)
    
    def _check_xs_values(self, datdir, nsims, ndims, ntimes):
        errors = set()
        data = [np.load(f"{datdir}/sim{i}/xs.npy", allow_pickle=True)
                for i in range(nsims)]
        simidx = 0
        tidx0 = 0
        tidx1 = 1
        for item in self.dataset:
            x0_exp = data[simidx][tidx0]
            x1_exp = data[simidx][tidx1]
            inputs, x1 = item
            x0 = inputs[1]
            if not np.allclose(x0, x0_exp):
                errors.add("Encountered incorrect values in x0")
            if not np.allclose(x1, x1_exp):
                errors.add("Encountered incorrect values in x1")
            tidx0 += 1
            tidx1 += 1
            if tidx1 == ntimes:
                simidx += 1
                tidx0 = 0
                tidx1 = 1
        return list(errors)

    @pytest.mark.parametrize(
        "transform, ncells_sample, expect_context, suppress_warnings", [
            # Warns that Dataset is not homogeneous
            [None, 0, pytest.warns(
                RuntimeWarning, match="Dataset is not homogeneous*"), False],
            # Tries to sample without setting ncells_sample
            ['sample', 0, pytest.raises(
                RuntimeError, match="Dataset is not homogeneous*"), False],
            # Specifies ncells_sample but does not set transform='sample'
            [None, 5, pytest.raises(
                RuntimeError, match="Must set transform='sample'*"), False],
            ['sample', 5, does_not_raise(), False],
            ['sample', 10, does_not_raise(), False],
            ['sample', 20, does_not_raise(), False],
    ])
    def test_length_and_shapes(
            self, datdir, nsims, ntimes, ndims, sigparamshape,
            transform, ncells_sample, expect_context, suppress_warnings,
    ):
        with expect_context:
            self._load_dataset(
                datdir, nsims, ndims, transform, ncells_sample,
                suppress_warnings=suppress_warnings,
            )
            errors = []
            errmsg = self._check_length(nsims * (ntimes - 1))
            if errmsg:
                errors.append(errmsg)
            shape_errors = self._check_item_shapes(
                (), sigparamshape, transform, ncells_sample,
            )
            errors += shape_errors
            assert not errors, "Errors occurred:\n{}".format("\n".join(errors))

    def test_values(
            self, datdir, nsims, ntimes, ndims, sigparamshape,
    ):
        with pytest.warns(RuntimeWarning):
            self._load_dataset(datdir, nsims, ndims, None, 0)
            errors = []
            xs_value_errors = self._check_xs_values(
                datdir, nsims, ndims, ntimes
            )
            errors += xs_value_errors
            assert not errors, "Errors occurred:\n{}".format("\n".join(errors))


@pytest.mark.parametrize("datdir, nsims, ntimes, ncells, ndims, sigparamshape", [
    [f"{datdir}/dataset1", 8, 11, 20, 2, (2,4)],
])
class TestHomogeneousDataloader:

    def _load_dataset(self, datdir, nsims, ndims, ncells_sample, batch_size):
        self.dataset = LandscapeSimulationDataset(
            datdir, 
            nsims=nsims,
            ndims=ndims,
            transform=None,
            ncells_sample=ncells_sample,
            seed=1,
        )
        self.dloader = NumpyLoader(
            self.dataset, 
            batch_size=batch_size, 
            shuffle=False,
        )
        return self.dataset, self.dloader
    
    def _check_length(self, batch_size):
        ndata = len(self.dataset)
        length_expected = ndata / batch_size
        if len(self.dloader) != length_expected:
            return f"Expected len {length_expected}. Got {len(self.dloader)}."
    
    def _check_item_shapes(
            self, shape_t_exp, shape_x_exp, shape_sigparams_exp
    ):
        errors = set()
        for bidx, batched_items in enumerate(self.dloader):
            if len(batched_items) != 2:
                msg = f"Encountered item in dataset not of length 2."
                errors.add(msg)
            else:
                inputs, outputs = batched_items
                # Check inputs should have 4 values with following shapes...
                if len(inputs) != 4:
                    msg = f"Encountered inputs in dataset not of length 4."
                    errors.add(msg)
                t0, x0, t1, sigparams = inputs
                x1 = outputs
                if t0.shape != shape_t_exp:
                    errors.add(f"Encountered incorrect shape of t0.")
                if t1.shape != shape_t_exp:
                    errors.add(f"Encountered incorrect shape of t1.")
                if x0.shape != shape_x_exp:
                    errors.add(f"Encountered incorrect shape of x0.")
                if x1.shape != shape_x_exp:
                    errors.add(f"Encountered incorrect shape of x1.")
                if sigparams.shape != shape_sigparams_exp:
                    errors.add(f"Encountered incorrect shape of sigparams.")
        return list(errors)
    
    @pytest.mark.parametrize('batch_size', [1, 2, 4])    
    def test_length_and_shapes(
            self, datdir, nsims, ntimes, ncells, ndims, sigparamshape, 
            batch_size
    ):
        self._load_dataset(datdir, nsims, ndims, 0, batch_size)
        errors = []
        errmsg = self._check_length(batch_size)
        if errmsg:
            errors.append(errmsg)
        shape_errors = self._check_item_shapes(
            (batch_size,), 
            (batch_size, ncells, ndims), 
            (batch_size, *sigparamshape)
        )
        errors += shape_errors
        assert not errors, "Errors occurred:\n{}".format("\n".join(errors))


@pytest.mark.parametrize(
    "datdir, nsims, ntimes, ndims, sigparamshape", [
    [f"{datdir}/dataset2", 8, 11, 2, (2, 4)],
])
@pytest.mark.parametrize(
    "transform, ncells_sample, expect_context, suppress_warnings", [
        # Tries to use NumpyLoader on Nonhomogeneous data. Also raises warning. 
        [None, 0, pytest.raises(RuntimeError, 
            match=NumpyLoader.nonloadable_error_message), True
        ],
        # Tries to sample without setting ncells_sample
        ['sample', 0, pytest.raises(
            RuntimeError, 
            match=LandscapeSimulationDataset.nonhomogeneous_warning), 
            False],
        # Specifies ncells_sample but does not set transform='sample'
        [None, 5, pytest.raises(
            RuntimeError, 
            match=LandscapeSimulationDataset.no_sample_with_ncells_warning), 
            False],
        ['sample', 5, does_not_raise(), False],
        ['sample', 10, does_not_raise(), False],
        ['sample', 20, does_not_raise(), False],
])
class TestNonhomogeneousDataloader:

    def _load_dataset(
            self, datdir, nsims, ndims, 
            transform, ncells_sample, batch_size,
            suppress_warnings=False
    ):
        self.dataset = LandscapeSimulationDataset(
            datdir, 
            nsims=nsims,
            ndims=ndims,
            transform=transform,
            ncells_sample=ncells_sample,
            seed=1,
            suppress_warnings=suppress_warnings,
        )
        self.dloader = NumpyLoader(
            self.dataset, 
            batch_size=batch_size, 
            shuffle=False,
        )
        return self.dataset, self.dloader
    
    def _check_length(self, batch_size):
        ndata = len(self.dataset)
        length_expected = ndata / batch_size
        if len(self.dloader) != length_expected:
            return f"Expected len {length_expected}. Got {len(self.dloader)}."
    
    def _check_item_shapes(
            self, shape_t_exp, shape_x_exp, shape_sigparams_exp
    ):
        errors = set()
        for bidx, batched_items in enumerate(self.dloader):
            if len(batched_items) != 2:
                msg = f"Encountered item in dataset not of length 2."
                errors.add(msg)
            else:
                inputs, outputs = batched_items
                # Check inputs should have 4 values with following shapes...
                if len(inputs) != 4:
                    msg = f"Encountered inputs in dataset not of length 4."
                    errors.add(msg)
                t0, x0, t1, sigparams = inputs
                x1 = outputs
                if t0.shape != shape_t_exp:
                    errors.add(f"Encountered incorrect shape of t0.")
                if t1.shape != shape_t_exp:
                    errors.add(f"Encountered incorrect shape of t1.")
                if x0.shape != shape_x_exp:
                    errors.add(f"Encountered incorrect shape of x0.")
                if x1.shape != shape_x_exp:
                    errors.add(f"Encountered incorrect shape of x1.")
                if sigparams.shape != shape_sigparams_exp:
                    errors.add(f"Encountered incorrect shape of sigparams.")
        return list(errors)
    
    @pytest.mark.parametrize('batch_size', [1, 2, 4])
    def test_length_and_shapes(
            self, datdir, nsims, ntimes, ndims, sigparamshape,
            transform, ncells_sample, expect_context, suppress_warnings,
            batch_size,
    ):
        with expect_context:
            self._load_dataset(
                datdir, nsims, ndims, transform, ncells_sample, batch_size,
                suppress_warnings=suppress_warnings,
            )
            errors = []
            errmsg = self._check_length(batch_size)
            if errmsg:
                errors.append(errmsg)
            shape_errors = self._check_item_shapes(
                (batch_size,), 
                (batch_size, ncells_sample, ndims), 
                (batch_size, *sigparamshape)
            )
            errors += shape_errors
            if errors:
                print(errors)
            assert not errors, "Errors occurred:\n{}".format("\n".join(errors))
