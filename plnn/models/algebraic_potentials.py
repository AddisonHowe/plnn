"""Algebraic Potential Functions

"""

from abc import abstractmethod
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import equinox as eqx


def get_phi_module_from_id(id, args={}):
    if id == "binary choice" or id == "phi1":
        return BinaryChoicePotential()
    elif id == "binary flip" or id == "phi2":
        return BinaryFlipPotential()
    elif id == "stitched" or id == "phi3":
        return StichedBinaryPotential()
    elif id == "quadratic" or id == "phiq":
        return QuadraticPotential(a=args.get('a', 1.), b=args.get('b', 1.))
    else:
        raise RuntimeError(f"Unknown Algebraic Potential ID: {id}")


class AbstractAlgebraicPotential(eqx.Module):
    
    ndims: int
    id: str

    def __init__(self, ndims, id):
        self.ndims = ndims
        self.id = id

    @abstractmethod
    def phi(self, x: Float[Array, "ndims"]) -> Float:
        raise NotImplementedError()
    
    def grad_phi(self, x: Float[Array, "ndims"]) -> Float[Array, "ndims"]:
        return jax.jacrev(self.phi)(x)
    
    def get_id(self) -> str:
        return self.id
    

############################
##  Algebraic Potentials  ##
############################

class BinaryChoicePotential(AbstractAlgebraicPotential):

    def __init__(self):
        super().__init__(ndims=2, id="binary choice")

    def phi(self, x: Float[Array, "2"]) -> Float:
        x, y = x
        x2 = x*x
        y2 = y*y
        return x2*x2 + y2*y2 + y2*y - 4*x2*y + y2
    
    def grad_phi(self, x: Float[Array, "2"]) -> Float[Array, "2"]:
        return jnp.array([
            4*x[0]**3 - 8*x[0]*x[1],
            4*x[1]**3 + 3*x[1]*x[1] - 4*x[0]*x[0] + 2*x[1]
        ])
    

class BinaryFlipPotential(AbstractAlgebraicPotential):

    def __init__(self):
        super().__init__(ndims=2, id="binary flip")

    def phi(self, x: Float[Array, "2"]) -> Float:
        x, y = x
        x2 = x*x
        y2 = y*y
        return x2*x2 + y2*y2 + x2*x - 2*x*y2 - x2
    
    def grad_phi(self, x: Float[Array, "2"]) -> Float[Array, "2"]:
        return jnp.array([
            4*x[0]**3 + 3*x[0]*x[0] - 2*x[1]*x[1] - 2*x[0],
            4*x[1]**3 - 4*x[0]*x[1]
        ])
    

class StichedBinaryPotential(AbstractAlgebraicPotential):

    r1: Float
    r2: Float

    def __init__(self, r1: Float=1., r2: Float=1.):
        super().__init__(ndims=2, id="stitched")
        self.r1 = r1
        self.r2 = r2

    def phi(self, x: Float[Array, "2"]) -> Float:
        x, y = x
        u, v = x - 2., y - 1.  # translation of x, y for F2
        c = 0.5 * (jnp.tanh(10.*(x - 0.5)) + 1.)  # chi transition function
        f1 = x**4 + y**4 + y**3 - 4*x*x*y + y*y
        f2 = u**4 + v**4 + u**3 - 2*u*v*v - u*u
        return self.r1 * (1. - c) * f1 + self.r2 * c * f2
    

class QuadraticPotential(AbstractAlgebraicPotential):

    a: Float[Array, "()"]
    b: Float[Array, "()"]

    def __init__(self, a=1., b=1.):
        super().__init__(ndims=2, id="quadratic")
        self.a = jnp.array(a)
        self.b = jnp.array(b)

    def phi(self, x: Float[Array, "2"]) -> Float:
        x, y = x
        return x*x*self.a + y*y*self.b
    
    def grad_phi(self, x: Float[Array, "2"]) -> Float[Array, "2"]:
        return jnp.array([
            2*self.a*x[0],
            2*self.b*x[1]
        ])
    