"""Non-Euclidean GMM Parameterized Landscape.

"""

import json
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
import equinox as eqx
from jaxtyping import Array, Float

from .plnn import _get_nn_init_args, _get_nn_init_func
from .ne_plnn import NEPLNN


class NEGMMPhiPLNN(NEPLNN):

    ncomponents: int

    def __init__(
            self, 
            key, dtype=jnp.float32, *,
            ncomponents,
            **kwargs
    ):
        key, subkey = jrandom.split(key, 2)
        super().__init__(subkey, dtype=dtype, **kwargs)

        self.model_type = "gmm_plnn"
        self.ncomponents = ncomponents

        key, subkey = jrandom.split(key, 2)
        self.phi_module = self._construct_phi_module(
            subkey, 
            dtype=dtype
        )
    
    ######################
    ##  Getter Methods  ##
    ######################

    def get_parameters(self) -> dict:
        """Return dictionary of learnable model parameters.

        The returned dictionary contains the following strings as keys: 
            phi.w, phi.b, tilt.w, tilt.b, metric.w, metric.b, sigma
        Each key maps to a list of jnp.array objects.
        
        Returns:
            dict: Dictionary containing learnable parameters.
        """
        d = super().get_parameters()
        def linear_layers(m):
            return [x for x in m.layers if isinstance(x, eqx.nn.Linear)]
        raise NotImplementedError()
        phi_linlayers  = linear_layers(self.phi_module)
        d.update({
            'phi.w' : [l.weight for l in phi_linlayers],
            'phi.b' : [l.bias for l in phi_linlayers],
        })
        return d
    
    def get_hyperparameters(self) -> dict:
        """Return dictionary of hyperparameters specifying the model.
        
        Returns:
            dict: dictionary of hyperparameters.
        """
        d = super().get_hyperparameters()
        d['ncomponents'] = self.ncomponents
        return d
    
    def get_linear_layer_parameters(self, include_metric=False) -> list[Array]:
        """Return a list of learnable parameters from linear layers.
        
        Args:
            include_metric (bool, optional) : Whether to include metric params.
                Default False.
        Returns:
            list[Array] : List of linear layer learnable parameter arrays.
        """
        params = super().get_linear_layer_parameters(include_metric)
        def linear_layers(m):
            return [x for x in m.layers if isinstance(x, eqx.nn.Linear)]
        raise NotImplementedError()
        phi_linlayers  = linear_layers(self.phi_module)
        phi_params = []
        for layer in phi_linlayers:
            phi_params.append(layer.weight)
            if layer.bias is not None:
                phi_params.append(layer.bias)
        return phi_params + params
    
    ##############################
    ##  Core Landscape Methods  ##
    ##############################

    @eqx.filter_jit
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
        raise NotImplementedError()
        return self.eval_confinement(y) + self.phi_module(y).squeeze(-1)

    @eqx.filter_jit
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
        raise NotImplementedError()
        return self.eval_grad_confinement(y) + \
               jax.jacrev(self.phi_module)(y).squeeze(0)

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
        sigma_init=1e-2, 
        solver='euler', 
        dt0=1e-2, 
        confine=False,
        sample_cells=True, 
        infer_metric=True,
        include_phi_bias=True, 
        include_tilt_bias=False,
        include_metric_bias=True,
        phi_hidden_dims=[16,32,32,16], 
        phi_hidden_acts='softplus', 
        phi_final_act=None, 
        phi_layer_normalize=False, 
        tilt_hidden_dims=[],
        tilt_hidden_acts=None,
        tilt_final_act=None,
        tilt_layer_normalize=False,
        metric_hidden_dims=[8,8,8,8], 
        metric_hidden_acts='softplus', 
        metric_final_act=None, 
        metric_layer_normalize=False, 
    ) -> tuple['NEGMMPhiPLNN', dict]:
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
            sigma_init
            solver
            dt0
            confine
            sample_cells
            infer_metric
            include_phi_bias
            include_tilt_bias
            include_metric_bias
            phi_hidden_dims
            phi_hidden_acts
            phi_final_act
            phi_layer_normalize
            tilt_hidden_dims
            tilt_hidden_acts
            tilt_final_act
            tilt_layer_normalize
            metric_hidden_dims
            metric_hidden_acts
            metric_final_act
            metric_layer_normalize
        
        Returns:
            NEGMMPhiPLNN: Model instance.
            dict: Dictionary of hyperparameters.
        """
        raise NotImplementedError()
        model = NEGMMPhiPLNN(
            key=key,
            dtype=dtype,
            ndims=ndims, 
            nparams=nparams, 
            nsigs=nsigs, 
            ncells=ncells,
            signal_type=signal_type, 
            nsigparams=nsigparams,
            sigma_init=sigma_init,
            solver=solver, 
            dt0=dt0, 
            confine=confine, 
            sample_cells=sample_cells,
            infer_metric=infer_metric,
            include_tilt_bias=include_tilt_bias,
            include_metric_bias=include_metric_bias,
            tilt_hidden_dims=tilt_hidden_dims, 
            tilt_hidden_acts=tilt_hidden_acts, 
            tilt_final_act=tilt_final_act,
            tilt_layer_normalize=tilt_layer_normalize,
            metric_hidden_dims=metric_hidden_dims, 
            metric_hidden_acts=metric_hidden_acts, 
            metric_final_act=metric_final_act,
            metric_layer_normalize=metric_layer_normalize,
        )
        hyperparams = model.get_hyperparameters()
        # Append to dictionary those hyperparams not stored internally.
        hyperparams.update({
            'tilt_hidden_dims' : tilt_hidden_dims,
            'tilt_hidden_acts' : tilt_hidden_acts,
            'tilt_final_act' : tilt_final_act,
            'tilt_layer_normalize' : tilt_layer_normalize,
            'metric_hidden_dims' : metric_hidden_dims,
            'metric_hidden_acts' : metric_hidden_acts,
            'metric_final_act' : metric_final_act,
            'metric_layer_normalize' : metric_layer_normalize,
        })
        model = jtu.tree_map(
            lambda x: x.astype(dtype) if eqx.is_array(x) else x, model
        )
        return model, hyperparams
    
    #############################
    ##  Initialization Method  ##
    #############################

    def initialize(self, 
            key, dtype=jnp.float32, *,
            init_tilt_weights_method='xavier_uniform',
            init_tilt_weights_args=[],
            init_tilt_bias_method='constant',
            init_tilt_bias_args=[0.],
            init_metric_weights_method='xavier_uniform',
            init_metric_weights_args=[],
            init_metric_bias_method='constant',
            init_metric_bias_args=[0.],
    ) -> 'NEGMMPhiPLNN':
        """Return an initialized version of the model.

        Args:
            key
            dtype
            init_phi_weights_method
            init_phi_weights_args
            init_phi_bias_method
            init_phi_bias_args
            init_tilt_weights_method
            init_tilt_weights_args
            init_tilt_bias_method
            init_tilt_bias_args
            init_metric_weights_method
            init_metric_weights_args
            init_metric_bias_method
            init_metric_bias_args

        Returns:
            NEGMMPhiPLNN : Initialized model instance.
        """        
        key, subkey = jrandom.split(key, 2)
        model = super().initialize(
            subkey, dtype=dtype, 
            init_tilt_weights_method=init_tilt_weights_method,
            init_tilt_weights_args=init_tilt_weights_args,
            init_tilt_bias_method=init_tilt_bias_method,
            init_tilt_bias_args=init_tilt_bias_args,
            init_metric_weights_method=init_metric_weights_method,
            init_metric_weights_args=init_metric_weights_args,
            init_metric_bias_method=init_metric_bias_method,
            init_metric_bias_args=init_metric_bias_args,
        )
        raise NotImplementedError()
        key, key1, key2 = jrandom.split(key, 3)
        is_linear = lambda x: isinstance(x, eqx.nn.Linear)

        return model

    ############################
    ##  Model Saving/Loading  ##
    ############################

    @staticmethod
    def load(fname:str, dtype=jnp.float32) -> tuple['NEGMMPhiPLNN', dict]:
        """Load a model from a binary parameter file.
        
        Args:
            fname (str): Parameter file to load.
            dtype : Datatype. Either jnp.float32 or jnp.float64.

        Returns:
            NEGMMPhiPLNN: Model instance.
            dict: Model hyperparameters.
        """
        with open(fname, "rb") as f:
            hyperparams = json.loads(f.readline().decode())
            model, _ = NEGMMPhiPLNN.make_model(
                key=jrandom.PRNGKey(0), dtype=dtype, **hyperparams
            )
            return eqx.tree_deserialise_leaves(f, model), hyperparams

    ######################
    ##  Helper Methods  ##
    ######################

    def _construct_phi_module(self, key, dtype=jnp.float32):
        raise NotImplementedError()
