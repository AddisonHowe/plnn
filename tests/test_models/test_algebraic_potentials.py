"""Tests for Algebraic Parameterized Landscapes.

"""

import pytest
import jax.numpy as jnp

from plnn.models.algebraic_potentials import AbstractAlgebraicPotential
from plnn.models.algebraic_potentials import get_phi_module_from_id

#####################
##  Configuration  ##
#####################

class NoGradBinaryChoicePotential(AbstractAlgebraicPotential):

    def __init__(self):
        super().__init__(ndims=2, id="binary choice nograd")

    def phi(self, x):
        x, y = x
        x2 = x*x
        y2 = y*y
        return x2*x2 + y2*y2 + y2*y - 4*x2*y + y2
    

class NoGradBinaryFlipPotential(AbstractAlgebraicPotential):

    def __init__(self):
        super().__init__(ndims=2, id="binary flip nograd")

    def phi(self, x):
        x, y = x
        x2 = x*x
        y2 = y*y
        return x2*x2 + y2*y2 + x2*x - 2*x*y2 - x2
        
def get_algebraic_potential(id):
    if id == "binary choice":
        return get_phi_module_from_id('phi1')
    elif id == "binary flip":
        return get_phi_module_from_id('phi2')
    elif id == "binary choice nograd":
        return NoGradBinaryChoicePotential()
    elif id == "binary flip nograd":
        return NoGradBinaryFlipPotential()


###############################################################################
###############################   BEGIN TESTS   ###############################
###############################################################################

@pytest.mark.parametrize('id', ['binary choice', 'binary choice nograd'])
@pytest.mark.parametrize('dtype, rtol, atol', [
    [jnp.float32, 1e-5, 1e-8],
    [jnp.float64, 1e-5, 1e-8],
])
class TestBinaryChoice:

    def _get_model(self, id):
        return get_algebraic_potential(id)
    
    @pytest.mark.parametrize('x, phi_exp', [
        [[0., 0.], 0.],
        [[0., 1.], 3.],
        [[1., 0.], 1.],
        [[1., 1.], 0.],
    ])
    def test_phi(self, id, dtype, rtol, atol, x, phi_exp):
        m = self._get_model(id)
        x = jnp.array(x, dtype=dtype)
        phi_exp = jnp.array(phi_exp, dtype=dtype)
        phi = m.phi(x)
        msg = f"Expected:\n{phi_exp}\nGot:\n{phi}"
        assert jnp.allclose(phi, phi_exp, rtol=rtol, atol=atol), msg
            

    @pytest.mark.parametrize('x, grad_phi_exp', [
        [[0., 0.], [0, 0]],
        [[0., 1.], [0, 9]],
        [[1., 0.], [4, -4]],
        [[1., 1.], [-4, 5]],
    ])
    def test_grad_phi(self, id, dtype, rtol, atol, x, grad_phi_exp):
        m = self._get_model(id)
        x = jnp.array(x, dtype=dtype)
        grad_phi_exp = jnp.array(grad_phi_exp, dtype=dtype)
        grad_phi = m.grad_phi(x)
        msg = f"Expected:\n{grad_phi_exp}\nGot:\n{grad_phi}"
        assert jnp.allclose(grad_phi, grad_phi_exp, rtol=rtol, atol=atol), msg
            


@pytest.mark.parametrize('id', ['binary flip', 'binary flip nograd'])
@pytest.mark.parametrize('dtype, rtol, atol', [
    [jnp.float32, 1e-5, 1e-8],
    [jnp.float64, 1e-5, 1e-8],
])
class TestBinaryFlip:

    def _get_model(self, id):
        return get_algebraic_potential(id)
    
    @pytest.mark.parametrize('x, phi_exp', [
        [[0., 0.], 0.],
        [[0., 1.], 1.],
        [[1., 0.], 1.],
        [[1., 1.], 0.],
    ])
    def test_phi(self, id, dtype, rtol, atol, x, phi_exp):
        m = self._get_model(id)
        x = jnp.array(x, dtype=dtype)
        phi_exp = jnp.array(phi_exp, dtype=dtype)
        phi = m.phi(x)
        msg = f"Expected:\n{phi_exp}\nGot:\n{phi}"
        assert jnp.allclose(phi, phi_exp, rtol=rtol, atol=atol), msg

    @pytest.mark.parametrize('x, grad_phi_exp', [
        [[0., 0.], [0, 0]],
        [[0., 1.], [-2, 4]],
        [[1., 0.], [5, 0]],
        [[1., 1.], [3, 0]],
    ])
    def test_grad_phi(self, id, dtype, rtol, atol, x, grad_phi_exp):
        m = self._get_model(id)
        x = jnp.array(x, dtype=dtype)
        grad_phi_exp = jnp.array(grad_phi_exp, dtype=dtype)
        grad_phi = m.grad_phi(x)
        msg = f"Expected:\n{grad_phi_exp}\nGot:\n{grad_phi}"
        assert jnp.allclose(grad_phi, grad_phi_exp, rtol=rtol, atol=atol), msg

