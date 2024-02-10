"""Model initialization tests.

"""

import pytest
from contextlib import nullcontext as does_not_raise
import numpy as np
import jax.numpy as jnp
import jax.random as jrandom

from plnn.models import DeepPhiPLNN, NEDeepPhiPLNN

#####################
##  Configuration  ##
#####################

def _make_model(model_class, dtype, **kwargs):
    nprng = np.random.default_rng(seed=None)
    key = jrandom.PRNGKey(nprng.integers(2**32))
    model, hp = model_class.make_model(key=key, dtype=dtype, **kwargs)
    return model, hp

def _init_model(model, dtype, **kwargs):
    key = jrandom.PRNGKey(1)
    model = model.initialize(key=key, dtype=dtype, **kwargs)
    return model

###############################################################################
###############################   BEGIN TESTS   ###############################
###############################################################################

@pytest.mark.parametrize('model_class', [DeepPhiPLNN, NEDeepPhiPLNN])
@pytest.mark.parametrize('dtype', [jnp.float32, jnp.float64])
class TestMake:
    
    def test_default_make(self, model_class, dtype):
        """Tests the default parameters of the make_model method."""
        key = jrandom.PRNGKey(0)
        model, _ = model_class.make_model(key, dtype=dtype)
        assert isinstance(model, model_class)

    @pytest.mark.parametrize("signal_type, nsigparams", [
        ['jump', 3],
        ['sigmoid', 4],
    ])
    def test_signal_args(self, model_class, dtype, signal_type, nsigparams):
        model, _ = _make_model(
            model_class, dtype,
            signal_type=signal_type, nsigparams=nsigparams
        )
        assert isinstance(model, model_class)

    @pytest.mark.parametrize("solver", ['euler', 'heun'])
    def test_solver(self, model_class, dtype, solver):
        model, _ = _make_model(
            model_class, dtype, solver=solver
        )
        assert isinstance(model, model_class)   


@pytest.mark.parametrize('model_class', [DeepPhiPLNN, NEDeepPhiPLNN])
@pytest.mark.parametrize('dtype', [jnp.float32, jnp.float64])
class TestPLNNInitializationLayerPhi:

    @pytest.mark.parametrize(
        "phi_hidden_dims, include_phi_bias, method, args, \
            expect_context, exp_shapes, exp_values", [
        [
            [2,3], True, 'constant',
            [0.],
            does_not_raise(), 
            [(2,2),(3,2),(1,3)],
            [np.zeros([2,2]), np.zeros([3,2]), np.zeros([1,3])],
        ],[
            [3], True, 'normal', 
            [1.],
            does_not_raise(), 
            [(3,2),(1,3)],
            None,
        ],[
            [], True, 'xavier_uniform', 
            [], 
            does_not_raise(), 
            [(1,2)],
            None,
        ],[
            [2,3], True, 'explicit',
            [[[[1,1],[1,1]], [[2,2],[2,2],[2,2]], [[3,3,3]]]],
            does_not_raise(), 
            [(2,2),(3,2),(1,3)],
            [np.ones([2,2]), 2*np.ones([3,2]), 3*np.ones([1,3])]
        ],[
            [2,3], True, 'explicit', 
            [[1,2,3]],
            does_not_raise(), 
            [(2,2),(3,2),(1,3)],
            [np.ones([2,2]), 2*np.ones([3,2]), 3*np.ones([1,3])]
        ],
    ])
    def test_phi_weights(
            self, 
            model_class, dtype, 
            phi_hidden_dims, include_phi_bias, 
            method, args, 
            expect_context, exp_shapes, exp_values,
    ):
        if exp_values:
            exp_values = [jnp.array(v, dtype=dtype) for v in exp_values]
        with expect_context:
            model, _ = _make_model(
                model_class, dtype,
                phi_hidden_dims=phi_hidden_dims,
                include_phi_bias=include_phi_bias,
            )
            model = _init_model(
                model, dtype, 
                init_phi_weights_method=method,
                init_phi_weights_args=args,
            )
            errors = []
            if not isinstance(model, model_class):
                msg = f"Wrong class. Expected {model_class}. Got {type(model)}"
                errors.append(msg)
            phi_ws = model.get_parameters()['phi.w']
            if len(phi_ws) != len(exp_shapes):
                msg = f"Got {len(phi_ws)} layers in phi.w. Expected {len(exp_shapes)}."
                errors.append(msg)
            for i, arr in enumerate(phi_ws):
                if not arr.shape == exp_shapes[i]:
                    msg = f"Wrong shape in phi layer {i}. Got {arr.shape}. "
                    msg += f" Expected {exp_shapes[i]}"
                    errors.append(msg)
                if exp_values and not jnp.allclose(phi_ws[i], exp_values[i]):
                    msg = f"Wrong values in phi layer {i}.\nGot:\n{phi_ws[i]}"
                    msg += f"\nExpected:\n{exp_values[i]}"
                    errors.append(msg)
            if errors:
                errors.append(f"phi.w:\n{phi_ws}")
            assert not errors, "Errors occurred:\n{}".format("\n".join(errors))


    @pytest.mark.parametrize(
        "phi_hidden_dims, include_phi_bias, method, args, \
            expect_context, exp_shapes, exp_values", [
        [
            [2,3], True, 'constant',
            [0.],
            does_not_raise(), 
            [(2,),(3,),(1,)],
            [np.zeros(2), np.zeros(3), np.zeros(1)],
        ],[
            [2,3], False, 'constant',
            [0.],
            does_not_raise(), 
            [(2,),(3,),(1,)],
            [None, None, None],
        ],[
            [3], True, 'normal', 
            [1.],
            does_not_raise(), 
            [(3,),(1,)],
            None,
        ],[
            [], True, 'xavier_uniform', 
            [], 
            pytest.raises(RuntimeError), 
            None,
            None,
        ],
        [
            [2,3], True, 'explicit',
            [[[1,1], [2,2,2], [3]]],
            does_not_raise(), 
            [(2,),(3,),(1,)],
            [np.ones(2), 2*np.ones(3), 3*np.ones(3)]
        ],[
            [2,3], True, 'explicit', 
            [[1,2,3]],
            does_not_raise(), 
            [(2,),(3,),(1,)],
            [np.ones(2), 2*np.ones(3), 3*np.ones(3)]
        ],
        
    ])
    def test_phi_bias(
            self, 
            model_class, dtype, 
            phi_hidden_dims, include_phi_bias, 
            method, args, 
            expect_context, exp_shapes, exp_values,
    ):
        if exp_values:
            exp_values = [None if v is None else jnp.array(v, dtype=dtype) 
                          for v in exp_values]
        with expect_context:
            model, _ = _make_model(
                model_class, dtype,
                phi_hidden_dims=phi_hidden_dims,
                include_phi_bias=include_phi_bias,
            )
            model = _init_model(
                model, dtype, 
                init_phi_bias_method=method,
                init_phi_bias_args=args,
            )
            errors = []
            if not isinstance(model, model_class):
                msg = f"Wrong class. Expected {model_class}. Got {type(model)}"
                errors.append(msg)
            phi_bs = model.get_parameters()['phi.b']
            if len(phi_bs) != len(exp_shapes):
                msg = f"Got {len(phi_bs)} layers in phi.b. Expected {len(exp_shapes)}."
                errors.append(msg)
            
            for i, arr in enumerate(phi_bs):
                if exp_values is None:
                    # skip value check
                    pass  
                elif exp_values[i] is None:
                    # check that the ith layer biases is None 
                    if arr is not None:
                        msg = f"Wrong values in phi layer {i}.\nGot:\n{phi_bs[i]}"
                        msg += f"\nExpected:\n{exp_values[i]}"
                        errors.append(msg)
                else:
                    # check the shape and values of the ith layer biases
                    if not arr.shape == exp_shapes[i]:
                        msg = f"Wrong shape in phi layer {i}. Got {arr.shape}. "
                        msg += f" Expected {exp_shapes[i]}"
                        errors.append(msg)
                    if exp_values and not jnp.allclose(phi_bs[i], exp_values[i]):
                        msg = f"Wrong values in phi layer {i}.\nGot:\n{phi_bs[i]}"
                        msg += f"\nExpected:\n{exp_values[i]}"
                        errors.append(msg)
            if errors:
                errors.append(f"phi.b:\n{phi_bs}")
            assert not errors, "Errors occurred:\n{}".format("\n".join(errors))

    # @pytest.mark.parametrize('method, args, expect_context', [
    #     ['constant', [0.], does_not_raise()],
    #     ['normal', [0., 1.], does_not_raise()],
    #     ['xavier_uniform', [], does_not_raise()],
    # ])
    # def test_tilt_weights(self, dtype, method, args, expect_context):
    #     with expect_context:
    #         model, _ = self._make_model()
    #         model = self._init_model(
    #             model, dtype, 
    #             init_tilt_weights_method=method,
    #             init_tilt_weights_args=args,
    #         )
    #         assert isinstance(model, DeepPhiPLNN)

    # @pytest.mark.parametrize('include_bias, method, args, expect_context', [
    #     [True, 'constant', [0.], does_not_raise()],
    #     [True, 'normal', [0., 1.], does_not_raise()],
    #     [True, 'xavier_uniform', [], pytest.raises(RuntimeError)],
    #     [False, 'constant', [0.], does_not_raise()],
    #     [False, 'normal', [0., 1.], does_not_raise()],
    # ])
    # def test_tilt_bias(self, dtype, include_bias, method, args, expect_context):
    #     with expect_context:
    #         model, _ = self._make_model(include_tilt_bias=include_bias)
    #         model = self._init_model(
    #             model, dtype, 
    #             init_tilt_bias_method=method,
    #             init_tilt_bias_args=args,
    #         )
    #         assert isinstance(model, DeepPhiPLNN)
