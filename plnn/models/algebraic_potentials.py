"""

"""

from abc import abstractmethod
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import equinox as eqx


def get_phi_module_from_id(id):
    if id == "binary choice" or id == "phi1":
        return BinaryChoicePotential()
    elif id == "binary flip" or id == "phi2":
        return BinaryFlipPotential()
    else:
        raise RuntimeError(f"Unknown Algebraic Potential ID: {id}")


class AbstractAlgebraicPotential(eqx.Module):
    
    ndims: int
    name: str

    def __init__(self, ndims, name):
        self.ndims = ndims
        self.name = name

    @abstractmethod
    def phi(self, x: Float[Array, "ndims"]) -> Float:
        raise NotImplementedError()
    
    def grad_phi(self, x: Float[Array, "ndims"]) -> Float[Array, "ndims"]:
        return jax.jacrev(self.phi)(x)
    
    def get_name(self) -> str:
        return self.name
    

############################
##  Algebraic Potentials  ##
############################

class BinaryChoicePotential(AbstractAlgebraicPotential):

    def __init__(self):
        super().__init__(ndims=2, name="binary choice")

    def phi(self, x: Float[Array, "2"]) -> Float:
        x, y = x
        x2 = x*x
        y2 = y*y
        return x2*x2 + y2*y2 + y2*y - 4*x2*y + y2
    
    def grad_phi(self, x: Float[Array, "2"]) -> Float[Array, "2"]:
        return jnp.array([
            4*x[0]**3 - 8*x[0]*x[1],
            4*x[1]**3 + 3*x[1]*x[1] - 4*x[0]*x[0] + 2*x[1]
        ]).T
    

class BinaryFlipPotential(AbstractAlgebraicPotential):

    def __init__(self):
        super().__init__(ndims=2, name="binary flip")

    def phi(self, x: Float[Array, "2"]) -> Float:
        x, y = x
        x2 = x*x
        y2 = y*y
        return x2*x2 + y2*y2 + x2*x - 2*x*y2 - x2
    
    def grad_phi(self, x: Float[Array, "2"]) -> Float[Array, "2"]:
        return jnp.array([
            4*x[0]**3 + 3*x[0]*x[0] - 2*x[1]*x[1] - 2*x[0],
            4*x[1]**3 - 4*x[0]*x[1]
        ]).T
    