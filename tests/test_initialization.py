import pytest
from contextlib import nullcontext as does_not_raise
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrandom

from plnn.models import PLNN, make_model, initialize_model


class TestMake:

    def _make_model(self, **kwargs):
        key = jrandom.PRNGKey(0)
        model, hyperparams = make_model(key=key, **kwargs)
        return model, hyperparams
    
    def test_default_make(self):
        key = jrandom.PRNGKey(0)
        model, _ = make_model(key)
        assert isinstance(model, PLNN)

    @pytest.mark.parametrize("signal_type, nsigparams", [['jump', 5]])
    def test_signal_args(self, signal_type, nsigparams):
        model, _ = self._make_model(
            signal_type=signal_type, nsigparams=nsigparams
        )
        assert isinstance(model, PLNN)

    @pytest.mark.parametrize("solver", ['euler'])
    def test_solver(self, solver):
        model, _ = self._make_model(solver=solver)
        assert isinstance(model, PLNN)   


@pytest.mark.parametrize('dtype', [jnp.float32, jnp.float64])
class TestInitialization:

    def _make_model(self, **kwargs):
        key = jrandom.PRNGKey(0)
        model, hyperparams = make_model(key=key, **kwargs)
        return model, hyperparams
    
    def _init_model(self, model, dtype, **kwargs):
        key = jrandom.PRNGKey(1)
        model = initialize_model(key, model, dtype, **kwargs)
        return model

    @pytest.mark.parametrize('method, args, expect', [
        ['constant', [0.], does_not_raise()],
        ['normal', [0., 1.], does_not_raise()],
        ['xavier_uniform', [], does_not_raise()],
        ['explicit', [[[1],[2],[3],[4],[5]]], does_not_raise()],
    ])
    def test_phi_weights(self, dtype, method, args, expect):
        with expect:
            model, _ = self._make_model()
            model = self._init_model(
                model, dtype, 
                init_phi_weights_method=method,
                init_phi_weights_args=args,
            )
            assert isinstance(model, PLNN)

    @pytest.mark.parametrize('include_bias, method, args, expect', [
        [True, 'constant', [0.], does_not_raise()],
        [True, 'normal', [0., 1.], does_not_raise()],
        [True, 'xavier_uniform', [], pytest.raises(RuntimeError)],
        [False, 'constant', [0.], does_not_raise()],
        [False, 'normal', [0., 1.], does_not_raise()],
    ])
    def test_phi_bias(self, dtype, include_bias, method, args, expect):
        with expect:
            model, _ = self._make_model(include_phi_bias=include_bias)
            model = self._init_model(
                model, dtype, 
                init_phi_bias_method=method,
                init_phi_bias_args=args,
            )
            assert isinstance(model, PLNN)

    @pytest.mark.parametrize('method, args, expect', [
        ['constant', [0.], does_not_raise()],
        ['normal', [0., 1.], does_not_raise()],
        ['xavier_uniform', [], does_not_raise()],
    ])
    def test_tilt_weights(self, dtype, method, args, expect):
        with expect:
            model, _ = self._make_model()
            model = self._init_model(
                model, dtype, 
                init_tilt_weights_method=method,
                init_tilt_weights_args=args,
            )
            assert isinstance(model, PLNN)

    @pytest.mark.parametrize('include_bias, method, args, expect', [
        [True, 'constant', [0.], does_not_raise()],
        [True, 'normal', [0., 1.], does_not_raise()],
        [True, 'xavier_uniform', [], pytest.raises(RuntimeError)],
        [False, 'constant', [0.], does_not_raise()],
        [False, 'normal', [0., 1.], does_not_raise()],
    ])
    def test_tilt_bias(self, dtype, include_bias, method, args, expect):
        with expect:
            model, _ = self._make_model(include_tilt_bias=include_bias)
            model = self._init_model(
                model, dtype, 
                init_tilt_bias_method=method,
                init_tilt_bias_args=args,
            )
            assert isinstance(model, PLNN)
