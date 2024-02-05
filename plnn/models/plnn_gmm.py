import json
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
import equinox as eqx
from jaxtyping import Array, Float

from .plnn import PLNN, _get_nn_init_args, _get_nn_init_func


class GMMPhiPLNN(PLNN):

    def __init__(self, key):
        super().__init__(key,)
        self.model_type = "gmm_plnn"
        raise NotImplementedError()
    
    ##############################
    ##  Core Landscape Methods  ##
    ##############################

    ######################
    ##  Getter Methods  ##
    ######################

    ##########################
    ##  Model Construction  ##
    ##########################
    
    #############################
    ##  Initialization Method  ##
    #############################

    ############################
    ##  Model Saving/Loading  ##
    ############################

    ###################################
    ##  Construction Helper Methods  ##
    ###################################
