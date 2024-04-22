"""Algebraic Parameterized Landscape Class.

"""

import json
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
from typing import Tuple
from jaxtyping import Array, Float
import equinox as eqx

from .plnn import PLNN
from .algebraic_potentials import AbstractAlgebraicPotential
from .algebraic_potentials import get_phi_module_from_id
    

class AlgebraicPL(PLNN):

    phi_module: AbstractAlgebraicPotential
    algebraic_phi_id: str
    tilt_module: eqx.Module

    def __init__(
            self, 
            key, dtype=jnp.float32, *,
            ndims=2,
            nparams=2,
            nsigs=2,
            sigma=1e-2,
            phi_module=None,
            algebraic_phi_id=None,
            tilt_weights=None,
            tilt_bias=None,
            include_tilt_bias=True,
            phi_args={},
            **kwargs
    ):
        key, subkey = jrandom.split(key, 2)
        super().__init__(
            subkey, dtype=dtype, 
            ndims=ndims, nsigs=nsigs, nparams=nparams, 
            confine=False, confinement_factor=1, sigma_init=sigma, 
            tilt_final_act=None, tilt_hidden_dims=[],
            include_tilt_bias=include_tilt_bias, 
            tilt_hidden_acts=None, 
            tilt_layer_normalize=False,
            **kwargs
        )
        self.model_type = "algebraic_plnn"
        
        # Initialize phi module
        if algebraic_phi_id is None:
            self.phi_module = phi_module
        else:
            self.phi_module = get_phi_module_from_id(
                algebraic_phi_id, args=phi_args
            )
        self.algebraic_phi_id = self.phi_module.get_id()
        
        # Initialize tilt module
        if tilt_weights is None:
            tilt_weights = jnp.zeros([self.ndims, self.nparams], dtype=dtype)
            tilt_weights[jnp.arange(self.nparams), jnp.arange(self.nparams)] = 1.
            assert tilt_weights.sum() == self.nparams
        else:
            tilt_weights = jnp.array(tilt_weights, dtype=dtype)
        
        if self.include_tilt_bias:
            if tilt_bias is None:
                tilt_bias = jnp.zeros(self.ndims, dtype=dtype)
            else:
                tilt_bias = jnp.array(tilt_bias, dtype=dtype)

        get_weights = lambda m: m.layers[0].weight
        get_bias = lambda m: m.layers[0].bias

        self.tilt_module = eqx.tree_at(get_weights, self.tilt_module, tilt_weights)

        if self.include_tilt_bias:
            self.tilt_module = eqx.tree_at(get_bias, self.tilt_module, tilt_bias)


    ######################
    ##  Getter Methods  ##
    ######################

    def get_parameters(self) -> dict:
        """Return dictionary of learnable model parameters.

        The returned dictionary contains the following strings as keys: 
            tilt.w, tilt.b, metric.w, metric.b, sigma
        Each key maps to a list of jnp.array objects.
        
        Returns:
            dict: Dictionary containing learnable parameters.
        """
        d = super().get_parameters()
        return d
    
    def get_hyperparameters(self) -> dict:
        """Return dictionary of hyperparameters specifying the model.
        
        Returns:
            dict: dictionary of hyperparameters.
        """
        d = super().get_hyperparameters()
        d['algebraic_phi_id'] = self.algebraic_phi_id
        return d
        
    def get_info_string(self, usetex=True) -> str:
        s = super().get_info_string(usetex=usetex)
        if not usetex:
            return s + f"\nAlgebraic Landscape ID: {self.algebraic_phi_id}"
        s += f"\\newline \\texttt{{{self.algebraic_phi_id}}}"
        s += f"\\newline $\\sigma={self.get_sigma():.3g}$"
        tw = self.get_parameters()['tilt.w'][0]
        tb = self.get_parameters()['tilt.b'][0]
        wstr = "\\\\".join(['&'.join([f"{x:.3g}" for x in row]) for row in tw])
        wstr = f"\\begin{{bmatrix}}{wstr}\\end{{bmatrix}}"
        if tb is None:
            bstr = ""
        else:
            bstr = "\\\\".join([f"{x:.3g}" for x in tb])
            bstr = f"+\\begin{{bmatrix}}{bstr}\\end{{bmatrix}}"
        s += f"\\newline $\\boldsymbol{{\\tau}}=\\Psi(\\boldsymbol{{s}})"
        s += f"={wstr}\\boldsymbol{{s}}{bstr}$"
        return s

    ##############################
    ##  Core Landscape Methods  ##
    ##############################
    
    def eval_phi(
        self, 
        y: Float[Array, "ndims"]
    ) -> Float:
        """Evaluate potential value without tilt.

        Args:
            y (Array) : State. Shape (d,).
        Returns:
            Array of shape (1,).
        """
        return self.phi_module.phi(y)

    def eval_grad_phi(
        self, 
        t: Float, 
        y: Float[Array, "ndims"]
    ) -> Float[Array, "ndims"]:
        """Evaluate gradient of potential without tilt.

        Args:
            t (Scalar) : Time.
            y (Array)  : State. Shape (d,).
        Returns:
            Array of shape (d,).
        """
        return self.phi_module.grad_phi(y)
    
    ##########################
    ##  Model Construction  ##
    ##########################

    @staticmethod
    def make_model(
        key, *, 
        dtype=jnp.float32,
        ndims=2, 
        nparams=2,
        nsigs=2, 
        ncells=100, 
        signal_type='jump', 
        nsigparams=3, 
        sigma=1e-2, 
        solver='euler', 
        dt0=1e-2, 
        sample_cells=True, 
        tilt_weights=None,
        include_tilt_bias=True,
        tilt_bias=None,
        phi_module=None,
        algebraic_phi_id=None,
        phi_args={},
    ) -> Tuple['AlgebraicPL', dict]:
        """Construct a model and store all hyperparameters.
        
        Args:
            key
            dtype
            ndims
            nparams
            nsigs
            ncells
            signal_type
            nsigparams
            sigma
            solver
            dt0
            confine
            confinement_factor
            sample_cells
            include_tilt_bias
            include_metric_bias
            tilt_hidden_dims
            tilt_hidden_acts
            tilt_final_act
            tilt_layer_normalize
        
        Returns:
            AlgebraicPL: Model instance.
            dict: Dictionary of hyperparameters.
        """
        model = AlgebraicPL(
            key=key,
            dtype=dtype,
            ndims=ndims, 
            nparams=nparams, 
            nsigs=nsigs, 
            ncells=ncells,
            signal_type=signal_type, 
            nsigparams=nsigparams,
            sigma=sigma,
            solver=solver, 
            dt0=dt0, 
            sample_cells=sample_cells,
            phi_module=phi_module,
            algebraic_phi_id=algebraic_phi_id,
            tilt_weights=tilt_weights,
            include_tilt_bias=include_tilt_bias,
            tilt_bias=tilt_bias,
            phi_args=phi_args,
        )
        hyperparams = model.get_hyperparameters()
        model = jtu.tree_map(
            lambda x: x.astype(dtype) if eqx.is_array(x) else x, model
        )
        return model, hyperparams

    #############################
    ##  Initialization Method  ##
    #############################
    
    def initialize(self, 
            key, dtype=jnp.float32, **kwargs
    ) -> 'AlgebraicPL':
        """Return an initialized version of the model.

        Returns:
            AlgebraicPL : Initialized model instance.
        """        
        key, subkey = jrandom.split(key, 2)
        model = super().initialize(
            subkey, dtype=dtype, **kwargs
        )
        return model
    
    ############################
    ##  Model Saving/Loading  ##
    ############################

    @staticmethod
    def load(fname:str, dtype=jnp.float32) -> tuple['AlgebraicPL', dict]:
        """Load a model from a binary parameter file.
        
        Args:
            fname (str): Parameter file to load.
            dtype : Datatype. Either jnp.float32 or jnp.float64.

        Returns:
            AlgebraicPL: Model instance.
            dict: Model hyperparameters.
        """
        with open(fname, "rb") as f:
            hyperparams = json.loads(f.readline().decode())
            model, _ = AlgebraicPL.make_model(
                key=jrandom.PRNGKey(0), dtype=dtype, **hyperparams
            )
            return eqx.tree_deserialise_leaves(f, model), hyperparams
