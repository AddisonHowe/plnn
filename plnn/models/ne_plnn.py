"""Abstract Base Class for a Non-Euclidean Parameterized Landscape.

"""

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, Float
import equinox as eqx

from .plnn import PLNN

##############################################################################
########################  Abstract NEPLNN Base Class  ########################
##############################################################################

class NEPLNN(PLNN):

    metric_module: eqx.Module  # learnable
    
    infer_metric: bool
    include_metric_bias: bool

    def __init__(
        self, 
        key, dtype=jnp.float32, *,
        infer_metric,
        include_metric_bias,
        metric_hidden_dims,
        metric_hidden_acts,
        metric_final_act,
        metric_layer_normalize,
        **kwargs
    ):
        key, subkey = jrandom.split(key, 2)
        super().__init__(subkey, dtype=dtype, **kwargs)
        
        self.infer_metric = infer_metric
        self.include_metric_bias = include_metric_bias

        key, key_metric = jax.random.split(key, 2)

        # Metric Neural Network: Maps ndims to (ndims, ndims).
        self.metric_module = self._construct_metric_module(
            key_metric, 
            metric_hidden_dims, 
            metric_hidden_acts, 
            metric_final_act, 
            metric_layer_normalize,
            bias=include_metric_bias, 
            dtype=dtype
        )

    ######################
    ##  Getter Methods  ##
    ######################
        
    def get_parameters(self) -> dict:
        """Return dictionary of learnable model parameters.

        The returned dictionary contains the following strings as keys: 
            metric.w, metric.b
        Each key maps to a list of jnp.array objects.
        
        Returns:
            dict: Dictionary containing learnable parameters.
        """
        d = super().get_parameters()
        metric_linlayers = self._get_linear_module_layers(self.metric_module)
        d.update({
            'metric.w' : [l.weight for l in metric_linlayers],
            'metric.b' : [l.bias for l in metric_linlayers],
        })
        return d
        
    def get_hyperparameters(self) -> dict:
        """Return dictionary of hyperparameters specifying the model.
        
        Returns:
            dict: dictionary of hyperparameters.
        """
        d = super().get_hyperparameters()
        d['infer_metric'] = self.infer_metric
        d['include_metric_bias'] = self.include_metric_bias
        return d
    
    def get_linear_layer_parameters(self) -> list[Array]:
        """Return a list of learnable parameters from linear layers.
        
        Args:
            include_metric (bool, optional) : Whether to include metric params.
                Default False.
        Returns:
            list[Array] : List of linear layer learnable parameter arrays.
        """
        params = super().get_linear_layer_parameters()
        metric_linlayers  = self._get_linear_module_layers(self.metric_module)
        metric_params = []
        for layer in metric_linlayers:
            metric_params.append(layer.weight)
            if layer.bias is not None:
                metric_params.append(layer.bias)
        return params + metric_params

    ##########################
    ##  Simulation Methods  ##
    ##########################

    def drift(
            self,
            t: Float, 
            y: Float[Array, "ndims"], 
            sigparams: Float[Array, "nsigs nsigparams"],
    ) -> Float[Array, "ndims"]:
        """Drift term used in `simulate_path` method.

        Args:
            t (Scalar)        : Time.
            y (Array)         : State. Shape (d,).
            sigparams (Array) : Signal parameters. Shape (nsigs, nsigparams).

        Returns:
            Array : Drift term. Shape (d,).
        """
        return self.eval_metric(t, y) @ self.eval_f(t, y, sigparams)
    
    def diffusion(
            self,
            t: Float, 
            y: Float[Array, "ndims"], 
    ) -> Float[Array, "ndims"]:
        """Diffusion term used in `simulate_path` method.

        Args:
            t (Scalar) : Time.
            y (Array)  : State. Shape (d,).
            args (Any) : Unused args placeholder.

        Returns:
            Array : Diffusion term. Shape (d,).
        """
        return self.eval_metric(t, y) @ self.eval_g(t, y)
        
    ##############################
    ##  Core Landscape Methods  ##
    ##############################
    
    def eval_metric(
        self,
        t: Float,
        y: Float[Array, "ndims"]
    ) -> Float[Array, "ndims ndims"]:
        """Evaluate metric tensor.
        
        Args:
            t (Scalar) : Time.
            y (Array)  : State. Shape (d,).
        Returns:
            Array of shape (d,d).
        """
        if self.infer_metric:
            # Get upper triangular values including diag.
            dm_vals = self.metric_module(y)
            dm = jnp.zeros([self.ndims, self.ndims])
            dm = dm.at[jnp.triu_indices(self.ndims)].set(dm_vals) # array
            dm = dm + dm.T
            dm = dm.at[jnp.diag_indices(self.ndims)].set(dm.diagonal() / 2)
        else:
            dm = 0
        return jnp.eye(self.ndims) + dm
    
    #############################
    ##  Initialization Method  ##
    #############################

    def initialize(
            self, 
            key, dtype=jnp.float32, *,
            init_metric_weights_method='xavier_uniform',
            init_metric_weights_args=[],
            init_metric_bias_method='constant',
            init_metric_bias_args=[0.],
            **kwargs
    ) -> 'NEPLNN':
        key, subkey = jrandom.split(key, 2)
        model = super().initialize(
            subkey, dtype=dtype, **kwargs
        )
        # Initialize metric module
        return self._initialize_linear_module(
            key, dtype, model, 'metric_module', 
            init_weights_method=init_metric_weights_method,
            init_weights_args=init_metric_weights_args,
            init_biases_method=init_metric_bias_method,
            init_biases_args=init_metric_bias_args,
            include_biases=self.include_metric_bias,
        )
    
    ###################################
    ##  Construction Helper Methods  ##
    ###################################
            
    def _construct_metric_module(self, key, hidden_dims, hidden_acts, final_act, 
                             layer_normalize, bias=True, dtype=jnp.float32):
        return self._construct_ffn(
            key, self.ndims, int(self.ndims * (self.ndims + 1) / 2), 
            hidden_dims, hidden_acts, final_act, layer_normalize, bias, dtype
        )
