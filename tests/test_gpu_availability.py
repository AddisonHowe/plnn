import pytest
import jax

@pytest.mark.use_gpu
def test_default_backend():
    assert jax.default_backend() == 'gpu'

@pytest.mark.use_gpu
def test_gpu_available():
    def jax_has_gpu():
        try:
            _ = jax.device_put(jax.numpy.ones(1), device=jax.devices('gpu')[0])
            return True
        except:
            return False
    assert jax_has_gpu()
    