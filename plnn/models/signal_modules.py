"""Signal Processing Modules

"""

from abc import abstractmethod
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import equinox as eqx


def get_psi_module_from_id(id):
    if id == "identity" or id == "id":
        return PsiIdentity
    else:
        raise RuntimeError(f"Unknown Signal Module ID: {id}")


class AbstractSignalModule(eqx.Module):

    nsigs: int
    nparams: int
    id: str

    def __init__(self, nsigs, nparams, id):
        self.nsigs = nsigs
        self.nparams = nparams
        self.id = id
        
    @abstractmethod
    def psi(self, signal, args):
        raise NotImplementedError()
    
    def get_id(self) -> str:
        return self.id


#################################
##  Signal Processing Modules  ##
#################################

class PsiIdentity(eqx.Module):

    def __init__(self, nsigs, nparams):
        super().__init__(nsigs, nparams, "identity")

    def psi(self, signal, args):
        return signal
    