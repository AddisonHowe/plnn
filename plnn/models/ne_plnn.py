"""Abstract Base Class for a Non-Euclidean Parameterized Landscape.

"""

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, Float
import equinox as eqx

from .plnn import PLNN, _get_nn_init_args, _get_nn_init_func

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
        def linear_layers(m):
            return [x for x in m.layers if isinstance(x, eqx.nn.Linear)]
        metric_linlayers = linear_layers(self.metric_module)
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
        def linear_layers(m):
            return [x for x in m.layers if isinstance(x, eqx.nn.Linear)]
        metric_linlayers  = linear_layers(self.metric_module)
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
    ) -> 'PLNN':
        if init_metric_bias_method == 'xavier_uniform':
            raise RuntimeError("Cannot initialize bias using `xavier_uniform`")
        
        key, subkey = jrandom.split(key, 2)
        model = super().initialize(
            subkey, dtype=dtype, 
            init_tilt_weights_method=init_tilt_weights_method,
            init_tilt_weights_args=init_tilt_weights_args,
            init_tilt_bias_method=init_tilt_bias_method,
            init_tilt_bias_args=init_tilt_bias_args,
            # init_metric_weights_method=init_metric_weights_method,
            # init_metric_weights_args=init_metric_weights_args,
            # init_metric_bias_method=init_metric_bias_method,
            # init_metric_bias_args=init_metric_bias_args,
        )
        model = self

        key, key1, key2, key3, key4 = jrandom.split(key, 5)
        is_linear = lambda x: isinstance(x, eqx.nn.Linear)
        
        # Initialize TiltNN Weights
        get_weights = lambda m: [
                x.weight 
                for x in jax.tree_util.tree_leaves(m.tilt_module, is_leaf=is_linear) 
                if is_linear(x)
            ]
        init_fn_args = _get_nn_init_args(init_tilt_weights_args)
        init_fn_handle = _get_nn_init_func(init_tilt_weights_method)
        if init_fn_handle:
            init_fn = init_fn_handle(*init_fn_args)
            weights = get_weights(model)
            new_weights = [
                init_fn(subkey, w.shape, dtype) 
                for w, subkey in zip(weights, jrandom.split(key1, len(weights)))
            ]
            model = eqx.tree_at(get_weights, model, new_weights)

        # Initialize TiltNN Bias if applicable
        get_biases = lambda m: [
                x.bias 
                for x in jax.tree_util.tree_leaves(m.tilt_module, is_leaf=is_linear) 
                if is_linear(x) and x.use_bias
            ]
        init_fn_args = _get_nn_init_args(init_tilt_bias_args)
        init_fn_handle = _get_nn_init_func(init_tilt_bias_method)
        if init_fn_handle and model.include_tilt_bias:
            init_fn = init_fn_handle(*init_fn_args)
            biases = get_biases(model)
            new_biases = [
                init_fn(subkey, b.shape, dtype) 
                for b, subkey in zip(biases, jrandom.split(key2, len(biases)))
            ]
            model = eqx.tree_at(get_biases, model, new_biases)

        # Initialize MetricNN Weights
        get_weights = lambda m: [
                x.weight 
                for x in jax.tree_util.tree_leaves(m.metric_module, is_leaf=is_linear) 
                if is_linear(x)
            ]
        init_fn_args = _get_nn_init_args(init_metric_weights_args)
        init_fn_handle = _get_nn_init_func(init_metric_weights_method)
        if init_fn_handle:
            init_fn = init_fn_handle(*init_fn_args)
            weights = get_weights(model)
            new_weights = [
                init_fn(subkey, w.shape, dtype) 
                for w, subkey in zip(weights, jrandom.split(key3, len(weights)))
            ]
            model = eqx.tree_at(get_weights, model, new_weights)

        # Initialize MetricNN Bias if applicable
        get_biases = lambda m: [
                x.bias 
                for x in jax.tree_util.tree_leaves(m.metric_module, is_leaf=is_linear) 
                if is_linear(x) and x.use_bias
            ]
        init_fn_args = _get_nn_init_args(init_metric_bias_args)
        init_fn_handle = _get_nn_init_func(init_metric_bias_method)
        if init_fn_handle and model.include_metric_bias:
            init_fn = init_fn_handle(*init_fn_args)
            biases = get_biases(model)
            new_biases = [
                init_fn(subkey, b.shape, dtype) 
                for b, subkey in zip(biases, jrandom.split(key4, len(biases)))
            ]
            model = eqx.tree_at(get_biases, model, new_biases)

        return model
    
    ###################################
    ##  Construction Helper Methods  ##
    ###################################
            
    def _construct_metric_module(self, key, hidden_dims, hidden_acts, final_act, 
                             layer_normalize, bias=True, dtype=jnp.float32):
        return self._construct_ffn(
            key, self.ndims, int(self.ndims * (self.ndims + 1) / 2), 
            hidden_dims, hidden_acts, final_act, layer_normalize, bias, dtype
        )
