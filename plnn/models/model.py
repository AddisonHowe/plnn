"""Landscape Potential Model

"""

import warnings
import json
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, Float
from diffrax import diffeqsolve, ControlTerm, MultiTerm, ODETerm, SaveAt
from diffrax import Euler, ReversibleHeun, VirtualBrownianTree
import equinox as eqx

_ACTIVATION_KEYS = {
    'none' : None,
    'softplus' : jax.nn.softplus,
    'elu' : jax.nn.elu,
    'tanh' : jax.nn.tanh,
}

def _explicit_initilizer(values):
    counter = 0
    def init_fn(key, shape, dtype):
        nonlocal counter
        v = jnp.array(values[counter], dtype=dtype)
        counter += 1
        return v
    return init_fn

_INIT_METHOD_KEYS = {
    'none' : None,
    'xavier_uniform' : jax.nn.initializers.glorot_uniform,  # no parameters
    'constant' : jax.nn.initializers.constant,  # parameters: constant value
    'normal' : jax.nn.initializers.normal,  # parameters: mean, std
    'explicit' : _explicit_initilizer, # parameters: values
}

class PLNN(eqx.Module):

    phi_nn: eqx.Module     # learnable
    tilt_nn: eqx.Module    # learnable
    logsigma: Array        # learnable
    metric_nn: eqx.Module  # learnable

    ndims: int
    nparams: int
    nsigs: int
    ncells: int
    signal_type: str
    nsigparams: int
    sigma_init: float
    solver: str
    dt0: float
    confine: bool
    sample_cells: bool
    infer_metric: bool
    include_phi_bias: bool
    include_tilt_bias: bool
    include_metric_bias: bool

    def __init__(
        self, 
        key,
        ndims,
        nparams,
        nsigs,
        ncells,
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
        phi_hidden_dims=[16, 32, 32, 16],  # end of attributes
        phi_hidden_acts='softplus',
        phi_final_act=None,
        phi_layer_normalize=False,
        tilt_hidden_dims=[],
        tilt_hidden_acts=None,
        tilt_final_act=None,
        tilt_layer_normalize=False,
        metric_hidden_dims=[8, 8, 8, 8],
        metric_hidden_acts='softplus',
        metric_final_act=None,
        metric_layer_normalize=False,
        dtype=jnp.float32,
    ):
        """
        """
        super().__init__()
        
        self.ndims = ndims
        self.nparams = nparams
        self.nsigs = nsigs
        self.ncells = ncells
        self.signal_type = signal_type
        self.nsigparams = nsigparams
        self.sigma_init = sigma_init
        self.solver = solver
        self.dt0 = dt0
        self.confine = confine
        self.sample_cells = sample_cells
        self.infer_metric = infer_metric
        self.include_phi_bias = include_phi_bias
        self.include_tilt_bias = include_tilt_bias
        self.include_metric_bias = include_metric_bias

        key, key1, key2, key3 = jax.random.split(key, 4)

        # Potential Neural Network: Maps ndims to a scalar.
        self.phi_nn = self._construct_phi_nn(
            key1, 
            phi_hidden_dims, 
            phi_hidden_acts, 
            phi_final_act, 
            phi_layer_normalize,
            bias=include_phi_bias, 
            dtype=dtype
        )
        
        # Tilt Neural Network: Linear tilt values. Maps nsigs to ndims.
        self.tilt_nn = self._construct_tilt_nn(
            key2, 
            tilt_hidden_dims,
            tilt_hidden_acts,
            tilt_final_act,
            tilt_layer_normalize,
            bias=include_tilt_bias, 
            dtype=dtype
        )

        # Metric Neural Network: Maps ndims to (ndims, ndims).
        self.metric_nn = self._construct_metric_nn(
            key3, 
            metric_hidden_dims, 
            metric_hidden_acts, 
            metric_final_act, 
            metric_layer_normalize,
            bias=include_metric_bias, 
            dtype=dtype
        )

        # Noise parameter
        self.logsigma = jnp.array(jnp.log(sigma_init), dtype=dtype)

    def __call__(
        self, 
        t0: Float[Array, "b"],
        t1: Float[Array, "b"],
        y0: Float[Array, "b ncells ndims"],
        sigparams: Float[Array, "b nsigs nsigparams"],
        key: Array,
    ) -> Float[Array, "b ncells ndims"]:
        """Forward call. Acts on batched data.

        TODO
        
        Args:
            t0 (Array) : Initial time.
            t1 (Array) : End time.
            y0 (Array) : Initial condition.
            sigparams (Array) : Signal parameters.
            dt (float) : Constant step size. Default 1e-3.

        Returns:
            Array of shape (n,d).
        """
        # Parse the inputs
        fwdvec = jax.vmap(self.simulate_forward, 0)
        key, sample_key = jrandom.split(key, 2)
        if self.sample_cells:
            y0 = self._sample_y0(sample_key, y0)
        batch_keys = jax.random.split(key, t0.shape[0])
        return fwdvec(t0, t1, y0, sigparams, batch_keys)
    
    ######################
    ##  Getter Methods  ##
    ######################
    
    def get_ncells(self):
        return self.ncells

    def get_sigma(self):
        return jnp.exp(self.logsigma.item())
    
    def get_parameters(self):
        """Return dictionary of learnable model parameters.

        Returned dictionary contains (str) keys: 
            phi.w, phi.b, tilt.w, tilt.b, sigma
        Each key maps to a list of jnp.array objects.
        
        Returns:
            dict: Dictionary containing learnable parameters.
                
        """
        def linear_layers(m):
            return [x for x in m.layers if isinstance(x, eqx.nn.Linear)]
        phi_linlayers  = linear_layers(self.phi_nn)
        tilt_linlayers = linear_layers(self.tilt_nn)
        metric_linlayers = linear_layers(self.metric_nn)
        d = {
            'phi.w'  : [l.weight for l in phi_linlayers],
            'phi.b'  : [l.bias for l in phi_linlayers],
            'tilt.w' : [l.weight for l in tilt_linlayers],
            'tilt.b' : [l.bias for l in tilt_linlayers],
            'metric.w' : [l.weight for l in metric_linlayers],
            'metric.b' : [l.bias for l in metric_linlayers],
            'sigma'  : self.get_sigma(),
        }
        return d
    
    def get_hyperparameters(self):
        """Return dictionary of hyperparameters specifying the model.
        
        Returns:
            dict: dictionary of hyperparameters.
        """
        return {
            'ndims' : self.ndims,
            'nparams' : self.nparams,
            'nsigs' : self.nsigs,
            'ncells' : self.ncells,
            'signal_type' : self.signal_type,
            'nsigparams' : self.nsigparams,
            'sigma_init' : self.sigma_init,
            'solver' : self.solver,
            'dt0' : self.dt0,
            'confine' : self.confine,
            'sample_cells' : self.sample_cells,
            'infer_metric' : self.infer_metric,
            'include_phi_bias' : self.include_phi_bias,
            'include_tilt_bias' : self.include_tilt_bias,
            'include_metric_bias' : self.include_metric_bias,
        }
    
    def get_linear_layer_parameters(self, include_metric=False):
        def linear_layers(m):
            return [x for x in m.layers if isinstance(x, eqx.nn.Linear)]
        phi_linlayers  = linear_layers(self.phi_nn)
        tilt_linlayers = linear_layers(self.tilt_nn)
        metric_linlayers = linear_layers(self.metric_nn)
        phi_params = []
        tilt_params = []
        metric_params = []
        for layer in phi_linlayers:
            phi_params.append(layer.weight)
            if layer.bias is not None:
                phi_params.append(layer.bias)
        for layer in tilt_linlayers:
            tilt_params.append(layer.weight)
            if layer.bias is not None:
                tilt_params.append(layer.bias)
        for layer in metric_linlayers:
            metric_params.append(layer.weight)
            if layer.bias is not None:
                metric_params.append(layer.bias)
        return phi_params + tilt_params \
                          + (metric_params if include_metric else [])

    ##########################
    ##  Simulation Methods  ##
    ##########################

    @eqx.filter_jit
    def simulate_forward(
        self,
        t0: Float,
        t1: Float,
        y0: Float[Array, "ncells ndims"],
        sigparams: Float[Array, "nsigs nsigparams"],
        key: Array,
    )->Float[Array, "ncells ndims"]:
        """Evolve forward in time using the Euler-Maruyama method.
        
        Args:
            TODO
        Returns:
            Array of shape (n,d).
        """
        subkeys = jrandom.split(key, len(y0))
        vecsim = jax.vmap(self.simulate_path, (None, None, 0, None, 0))
        return vecsim(t0, t1, y0, sigparams, subkeys).squeeze(1)
    
    @eqx.filter_jit
    def simulate_path(
        self,
        t0: Float,
        t1: Float,
        y0: Float[Array, "ndims"],
        sigparams: Float[Array, "nsigs nsigparams"],
        key: Array,
    ):
        """TODO
        """
        drift = lambda t, y, args: self.eval_metric(t, y) @ self.eval_f(t, y, sigparams)
        diffusion = lambda t, y, args: self.eval_metric(t, y) @ self.eval_g(t, y)
        brownian_motion = VirtualBrownianTree(
            t0, t1, tol=1e-3, 
            shape=(len(y0),), 
            key=key
        )
        terms = MultiTerm(
            ODETerm(drift), 
            ControlTerm(diffusion, brownian_motion)
        )
        solver = Euler()
        saveat = SaveAt(t1=True)
        sol = diffeqsolve(
            terms, solver, 
            t0, t1, dt0=self.dt0, 
            y0=y0, 
            saveat=saveat
        )
        return sol.ys
    
    ##############################
    ##  Core Landscape Methods  ##
    ##############################

    @eqx.filter_jit
    def eval_f(
        self, 
        t: Float, 
        y: Float[Array, "ndims"], 
        sigparams: Float[Array, "nsigs nsigparams"],
    ) -> Float[Array, "ndims"]:
        """Evaluate drift term. 

        Args:
            t (Scalar)         : Time.
            y (Array)          : State. Shape (d,).
            sigparams (Array) : Signal parameters. Shape (nsigs, nsigparams).
        Returns:
            Array of shape (d,).
        """
        gphi = self.eval_grad_phi(t, y)
        gtilt = self.grad_tilt(t, sigparams)
        return -(gphi + gtilt)

    @eqx.filter_jit
    def eval_g(
        self, 
        t: Float, 
        y: Float[Array, "ndims"]
    ) -> Float[Array, "ndims"]:
        """Evaluate diffusion term. 
        
        Currently only implements scalar noise. (TODO: generalize noise.)

        Args:
            t (Scalar) : Time.
            y (Array)  : State. Shape (d,).
        Returns:
            Array of shape (d,).
        """
        return jnp.exp(self.logsigma) * jnp.ones(y.shape)
    
    @eqx.filter_jit
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
            dm_vals = self.metric_nn(y)
            dm = jnp.zeros([self.ndims, self.ndims])
            dm = dm.at[jnp.triu_indices(self.ndims)].set(dm_vals) # array
            dm = dm + dm.T
            dm = dm.at[jnp.diag_indices(self.ndims)].set(dm.diagonal() / 2)
        else:
            dm = 0
        return jnp.eye(self.ndims) + dm
    
    @eqx.filter_jit
    def eval_confinement(
        self,
        y: Float[Array, "ndims"]
    ) -> Float:
        """Evaluate confinement term.
        
        Args:
            y (Array) : State. Shape (d,).
        Returns:
            Array of shape (1,).
        """
        if self.confine:
            return jnp.sum(jnp.power(y, 4))
        return 0.

    @eqx.filter_jit
    def eval_grad_confinement(
        self,
        y: Float[Array, "ndims"]
    ) -> Float[Array, "ndims"]:
        """Evaluate the gradient of the confinement term.
        
        Args:
            y (Array) : State. Shape (d,).
        Returns:
            Array of shape (d,).
        """
        if self.confine:
            return 4 * jnp.power(y, 3)
        return y * 0.

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
        return self.eval_confinement(y) + self.phi_nn(y).squeeze(-1)

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
        return self.eval_grad_confinement(y) + \
               jax.jacrev(self.phi_nn)(y).squeeze(0)
    
    @eqx.filter_jit
    def grad_tilt(
        self, 
        t: Float, 
        sigparams: Float[Array, "nsigs nsigparams"]
    ) -> Float[Array, "ndims"]:
        """Evaluate gradient of linear tilt function. 

        Args:
            t (Scalar) : Time.
            sigparams (Array) : Signal parameters. Shape (nsigs, nsigparams).
        Returns:
            Array of shape (ndims,).
        """
        if self.signal_type == "jump":
            signal_vals = self.binary_signal_function(t, sigparams)
        elif self.signal_type == "sigmoid":
            signal_vals = self.sigmoid_signal_function(t, sigparams)
        return self.tilt_nn(signal_vals)
    
    @eqx.filter_jit
    def eval_tilted_phi(
        self, 
        t: Float, 
        y: Float[Array, "ndims"], 
        sigparams: Float[Array, "nsigs nsigparams"],
    ) -> Float:
        """Evaluate tilted landscape level. 

        Args:
            t (Scalar)         : Time.
            y (Array)          : State. Shape (d,).
            sigparams (Array) : Signal parameters. Shape (nsigs, nsigparams).
        Returns:
            Array of shape (1,).
        """
        phi = self.eval_phi(y)
        gtilt = self.grad_tilt(t, sigparams)
        return phi + jnp.dot(gtilt, y)
    
    ####################################
    ##  Vectorized Landscape Methods  ##
    ####################################
    
    @eqx.filter_jit
    def f(
        self, 
        t: Float, 
        y: Float[Array, "ncells ndims"], 
        sigparams: Float[Array, "nsigs nsigparams"],
    ) -> Float[Array, "ncells ndims"]:
        """Drift term vectorized across a set of cells.

        Args:
            t (Scalar)         : Time.
            y (Array)          : State. Shape (n,d).
            sigparams (Array) : Signal parameters. Shape (nsigs, nsigparams).
        Returns:
            Array of shape (d,).
        """
        return jax.vmap(self.eval_f, (None, 0, None))(t, y, sigparams)

    @eqx.filter_jit
    def g(
        self, 
        t: Float, 
        y: Float[Array, "ncells ndims"]
    ) -> Float[Array, "ncells ndims"]:
        """Diffusion term vectorized across a set of cells. 

        Args:
            t (Scalar) : Time.
            y (Array)  : State. Shape (n,d).
        Returns:
            Array of shape (n,d).
        """
        return jax.vmap(self.eval_g, (None, 0))(t, y)
    
    @eqx.filter_jit
    def phi(
        self, 
        y: Float[Array, "ncells ndims"]
    ) -> Float[Array, "ncells"]:
        """Potential value without tilt, vectorized across a set of cells.

        Args:
            y (Array) : State. Shape (n,d).
        Returns:
            Array of shape (n,1).
        """
        return jax.vmap(self.eval_phi, 0)(y)

    @eqx.filter_jit
    def grad_phi(
        self, 
        t: Float, 
        y: Float[Array, "ncells ndims"]
    ) -> Float[Array, "ncells ndims"]:
        """Gradient of potential without tilt, vectorized across a set of cells.

        Args:
            t (Scalar) : Time.
            y (Array)  : State. Shape (n,d).
        Returns:
            Array of shape (n,d).
        """
        return jax.vmap(self.eval_grad_phi, (None, 0))(t, y)
    
    @eqx.filter_jit
    def tilted_phi(
        self, 
        t: Float, 
        y: Float[Array, "ncells ndims"],
        sigparams: Float[Array, "nsigs nsigparams"]
    ) -> Float[Array, "ncells"]:
        """Tilted landscape level. 

        Args:
            t (Scalar) : Time.
            y (Array)  : State. Shape (n,d).
            sigparams (Array) : Signal parameters. Shape (nsigs, nsigparams).
        Returns:
            Array of shape (n,1).
        """
        return jax.vmap(self.eval_tilted_phi, (None, 0, None))(t, y, sigparams)
        
    ########################
    ##  Signal Functions  ##
    ########################

    @eqx.filter_jit
    def binary_signal_function(
        self, 
        t: Float, 
        sigparams: Float[Array, "nsigs nsigparams"]
    ) -> Float[Array, "nsigs"]:
        """Evaluates a binary signal defined by given parameters at time t.

        Args:
            t (Scalar) : Time.
            sigparams (Array) : Signal parameters. Shape (nparams, nsigparams)
        Returns:
            Array of shape (nsigs,)
        """
        tcrit = sigparams[...,0]
        p0 = sigparams[...,1]
        p1 = sigparams[...,2]
        val = (t < tcrit) * p0 + (t >= tcrit) * p1
        return val
    
    @eqx.filter_jit
    def sigmoid_signal_function(
        self, 
        t: Float, 
        sigparams: Float[Array, "nsigs nsigparams"]
    ) -> Float[Array, "nsigs"]:
        """Evaluates a sigmoidal signal defined by given parameters at time t.

        Args:
            t (Scalar) : Time.
            sigparams (Array) : Signal parameters. Shape (nparams, nsigparams)
        Returns:
            Array of shape (nsigs,)
        """
        tcrit = sigparams[...,0]
        p0 = sigparams[...,1]
        p1 = sigparams[...,2]
        r = sigparams[...,3]
        val = p0 + 0.5*(p1 - p0) * (1 + jnp.tanh(r * (t - tcrit)))
        return val

    #############################
    ##  Initialization Method  ##
    #############################

    def initialize(self, key, dtype=jnp.float32, *,
        init_phi_weights_method='xavier_uniform',
        init_phi_weights_args=[],
        init_phi_bias_method='constant',
        init_phi_bias_args=[0.],
        init_tilt_weights_method='xavier_uniform',
        init_tilt_weights_args=[],
        init_tilt_bias_method='constant',
        init_tilt_bias_args=[0.],
        init_metric_weights_method='xavier_uniform',
        init_metric_weights_args=[],
        init_metric_bias_method='constant',
        init_metric_bias_args=[0.],
    ):
        new_model = initialize_model(
            key, self, dtype=dtype,
            init_phi_weights_method=init_phi_weights_method,
            init_phi_weights_args=init_phi_weights_args,
            init_phi_bias_method=init_phi_bias_method,
            init_phi_bias_args=init_phi_bias_args,
            init_tilt_weights_method=init_tilt_weights_method,
            init_tilt_weights_args=init_tilt_weights_args,
            init_tilt_bias_method=init_tilt_bias_method,
            init_tilt_bias_args=init_tilt_bias_args,
            init_metric_weights_method=init_metric_weights_method,
            init_metric_weights_args=init_metric_weights_args,
            init_metric_bias_method=init_metric_bias_method,
            init_metric_bias_args=init_metric_bias_args,
        )
        return new_model
    
    ###################################
    ##  Construction Helper Methods  ##
    ###################################

    def _check_hidden_layers(self, hidden_dims, hidden_acts, final_act):
        """Check the model architecture.
        """
        if hidden_dims is None \
                or (isinstance(hidden_dims, int) and hidden_dims <= 0) \
                or (isinstance(hidden_dims, list) and 0 in hidden_dims):
            hidden_dims = []
        nlayers = len(hidden_dims)
        # Convert singular hidden activation to a list.
        if hidden_acts is None or isinstance(hidden_acts, (str, type)):
            hidden_acts = [hidden_acts] * nlayers
        elif isinstance(hidden_acts, list):
            if len(hidden_acts) == 1:
                hidden_acts = hidden_acts * nlayers
            else:
                hidden_acts = hidden_acts.copy()  # avoid overwrite of original
        # Check number of hidden activations
        if len(hidden_acts) != nlayers:
            msg = "Number of activation functions must match number of " + \
                  f"hidden layers. Got {hidden_acts} for {nlayers} layers."
            raise RuntimeError(msg)
        # Check hidden activation types
        for i, val in enumerate(hidden_acts):
            if isinstance(val, str):
                hidden_acts[i] = _ACTIVATION_KEYS[val.lower()]
            elif isinstance(val, type) or val is None:
                pass
            else:
                msg = f"Cannot handle activation specs: {hidden_acts}"
                raise RuntimeError(msg)
        # Check final activation types
        if isinstance(final_act, str):
            final_act = _ACTIVATION_KEYS[final_act.lower()]
        elif isinstance(final_act, type) or final_act is None:
                pass
        else:
            msg = f"Cannot handle final activation spec: {final_act}"
            raise RuntimeError(msg)
        return hidden_dims, hidden_acts, final_act
    
    def _add_layer(self, key, layer_list, din, dout, 
                   activation, normalization, bias=True, dtype=jnp.float32):
        layer_list.append(eqx.nn.Linear(din, dout, key=key, use_bias=bias))
        if normalization:
            layer_list.append(eqx.nn.LayerNorm([dout]))  # TODO: untested
        if activation:
            layer_list.append(eqx.nn.Lambda(activation))
        
    def _construct_phi_nn(self, key, hidden_dims, hidden_acts, final_act, 
                          layer_normalize, bias=True, dtype=jnp.float32):
        return self._construct_ffn(
            key, self.ndims, 1,
            hidden_dims, hidden_acts, final_act, layer_normalize, bias, dtype
        )
    
    def _construct_tilt_nn(self, key, hidden_dims, hidden_acts, final_act, 
                          layer_normalize, bias=False, dtype=jnp.float32):
        return self._construct_ffn(
            key, self.nsigs, self.ndims, 
            hidden_dims, hidden_acts, final_act, layer_normalize, bias, dtype
        )
    
    def _construct_metric_nn(self, key, hidden_dims, hidden_acts, final_act, 
                             layer_normalize, bias=True, dtype=jnp.float32):
        return self._construct_ffn(
            key, self.ndims, int(self.ndims * (self.ndims + 1) / 2), 
            hidden_dims, hidden_acts, final_act, layer_normalize, bias, dtype
        )
    
    def _construct_ffn(self, key, dim0, dim1, hidden_dims, hidden_acts, 
                       final_act, layer_normalize, bias, dtype):
        hidden_dims, hidden_acts, final_act = self._check_hidden_layers(
            hidden_dims=hidden_dims, 
            hidden_acts=hidden_acts, 
            final_act=final_act,
        )

        layer_list = []
        
        if len(hidden_dims) == 0:
            self._add_layer(
                key, 
                layer_list, dim0, dim1, 
                activation=final_act, 
                normalization=layer_normalize,
                bias=bias,
                dtype=dtype,
            )
            return eqx.nn.Sequential(layer_list)
        
        # Hidden layers
        key, subkey = jrandom.split(key, 2)
        self._add_layer(
            subkey, 
            layer_list, dim0, hidden_dims[0], 
            activation=hidden_acts[0], 
            normalization=layer_normalize,
            bias=bias,
            dtype=dtype,
        )
        for i in range(len(hidden_dims) - 1):
            key, subkey = jrandom.split(key, 2)
            self._add_layer(
                subkey,
                layer_list, hidden_dims[i], hidden_dims[i+1], 
                activation=hidden_acts[i+1], 
                normalization=layer_normalize,
                bias=bias,
                dtype=dtype,
            )
        # Final layer
        key, subkey = jrandom.split(key, 2)
        self._add_layer(
            subkey,
            layer_list, hidden_dims[-1], dim1,
            activation=final_act, 
            normalization=False,
            bias=bias,
            dtype=dtype,
        )
        return eqx.nn.Sequential(layer_list)
    
    ########################
    ##  Plotting Methods  ##
    ########################

    def plot_phi(self, signal=None, r=4, res=50, plot3d=False, **kwargs):
        """Plot the scalar function phi.
        Args:
            r (int) : 
            res (int) :
            plot3d (bool) :
            normalize (bool) :
            log_normalize (bool) :
            clip (float) :
            ax (Axis) :
            figsize (tuple[float]) :
            xlims (tuple[float]) :
            ylims (tuple[float]) :
            xlabel (str) :
            ylabel (str) :
            zlabel (str) :
            title (str) :
            cmap (Colormap) :
            include_cbar (bool) :
            cbar_title (str) :
            cbar_titlefontsize (int) :
            cbar_ticklabelsize (int) :
            view_init (tuple) :
            saveas (str) :
            show (bool) :
        Returns:
            Axis object.
        """
        #~~~~~~~~~~~~  process kwargs  ~~~~~~~~~~~~#
        normalize = kwargs.get('normalize', True)
        log_normalize = kwargs.get('log_normalize', True)
        clip = kwargs.get('clip', None)
        ax = kwargs.get('ax', None)
        figsize = kwargs.get('figsize', (6, 4))
        xlims = kwargs.get('xlims', None)
        ylims = kwargs.get('ylims', None)
        xlabel = kwargs.get('xlabel', "$x$")
        ylabel = kwargs.get('ylabel', "$y$")
        zlabel = kwargs.get('zlabel', "$\\phi$")
        title = kwargs.get('title', "$\\phi(x,y)$")
        cmap = kwargs.get('cmap', 'coolwarm')
        include_cbar = kwargs.get('include_cbar', True)
        cbar_title = kwargs.get('cbar_title', 
                                "$\\ln\\phi$" if log_normalize else "$\\phi$")
        cbar_titlefontsize = kwargs.get('cbar_titlefontsize', 10)
        cbar_ticklabelsize = kwargs.get('cbar_ticklabelsize', 8)
        view_init = kwargs.get('view_init', (30, -45))
        saveas = kwargs.get('saveas', None)
        show = kwargs.get('show', False)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        if ax is None and plot3d:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        elif ax is None and (not plot3d):
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Handle signal value
        eval_time = 0.
        if signal is None:
            if self.signal_type == 'jump':
                signal = self.nsigs * [[1, 0, 0]]
            elif self.signal_type == 'sigmoid':
                signal = self.nsigs * [[1, 0, 0, 0]]
        signal_params = jnp.array(signal)

        # Compute phi
        x = np.linspace(-r, r, res)
        y = np.linspace(-r, r, res)
        xs, ys = np.meshgrid(x, y)
        z = np.array([xs.flatten(), ys.flatten()]).T
        z = jnp.array(z, dtype=jnp.float32)
        phi = np.array(self.tilted_phi(eval_time, z, signal_params))
        
        # Normalization
        if normalize:
            phi = 1 + phi - phi.min()  # set minimum to 1
        if log_normalize:
            phi = np.log(phi)

        # Clipping
        clip = 1 + phi.max() if clip is None else clip
        if clip < phi.min():
            warnings.warn(f"Clip value {clip} is below minimum value to plot.")
            clip = phi.max()
        under_cutoff = phi <= clip
        plot_screen = np.ones(under_cutoff.shape)
        plot_screen[~under_cutoff] = np.nan
        phi_plot = phi * plot_screen

        # Plot phi
        if plot3d:
            sc = ax.plot_surface(
                xs, ys, phi_plot.reshape(xs.shape), 
                vmin=phi[under_cutoff].min(),
                vmax=phi[under_cutoff].max(),
                cmap=cmap
            )
        else:
            sc = ax.pcolormesh(
                xs, ys, phi_plot.reshape(xs.shape),
                vmin=phi[under_cutoff].min(),
                vmax=phi[under_cutoff].max(),
                cmap=cmap, 
            )
        # Colorbar
        if include_cbar:
            cbar = plt.colorbar(sc)
            cbar.ax.set_title(cbar_title, size=cbar_titlefontsize)
            cbar.ax.tick_params(labelsize=cbar_ticklabelsize)
        
        # Format plot
        if xlims is not None: ax.set_xlim(*xlims)
        if ylims is not None: ax.set_ylim(*ylims)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if plot3d: 
            ax.set_zlabel(zlabel)
            ax.view_init(*view_init)
        plt.tight_layout()
        
        # Save and close
        if saveas: plt.savefig(saveas, bbox_inches='tight')
        if not show: plt.close()
        return ax
    
    def plot_f(self, signal=0, r=4, res=50, **kwargs):
        """Plot the vector field f.
        Args:
            signal (float or tuple[float]) :
            r (int) : 
            res (int) :
            ax (Axis) :
            figsize (tuple[float]) :
            xlims (tuple[float]) :
            ylims (tuple[float]) :
            xlabel (str) :
            ylabel (str) :
            title (str) :
            cmap (Colormap) :
            include_cbar (bool) :
            cbar_title (str) :
            cbar_titlefontsize (int) :
            cbar_ticklabelsize (int) :
            saveas (str) :
            show (bool) :
        Returns:
            Axis object.
        """
        #~~~~~~~~~~~~  process kwargs  ~~~~~~~~~~~~#
        ax = kwargs.get('ax', None)
        figsize = kwargs.get('figsize', (6, 4))
        xlims = kwargs.get('xlims', None)
        ylims = kwargs.get('ylims', None)
        xlabel = kwargs.get('xlabel', "$x$")
        ylabel = kwargs.get('ylabel', "$y$")
        title = kwargs.get('title', "$f(x,y|\\vec{s})$")
        cmap = kwargs.get('cmap', 'coolwarm')
        include_cbar = kwargs.get('include_cbar', True)
        cbar_title = kwargs.get('cbar_title', "$|f|$")
        cbar_titlefontsize = kwargs.get('cbar_titlefontsize', 10)
        cbar_ticklabelsize = kwargs.get('cbar_ticklabelsize', 8)
        saveas = kwargs.get('saveas', None)
        show = kwargs.get('show', False)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        if ax is None: fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Handle signal value
        eval_time = 0.
        if signal is None:
            if self.signal_type == 'jump':
                signal = [1, 0, 0]
            elif self.signal_type == 'sigmoid':
                signal = [1, 0, 0, 0]
        signal_params = jnp.tile(jnp.array(signal), (self.nsigs, 1))
        
        # Compute f
        x = np.linspace(-r, r, res)
        y = np.linspace(-r, r, res)
        xs, ys = np.meshgrid(x, y)
        z = np.array([xs.flatten(), ys.flatten()]).T
        z = jnp.array(z)
        f = np.array(self.f(eval_time, z, signal_params))
        fu, fv = f.T
        fnorms = np.sqrt(fu**2 + fv**2)

        # Plot force field, tilted by signals
        sc = ax.quiver(xs, ys, fu/fnorms, fv/fnorms, fnorms, cmap=cmap)
        
        # Colorbar
        if include_cbar:
            cbar = plt.colorbar(sc)
            cbar.ax.set_title(cbar_title, size=cbar_titlefontsize)
            cbar.ax.tick_params(labelsize=cbar_ticklabelsize)
        
        # Format plot
        if xlims is not None: ax.set_xlim(*xlims)
        if ylims is not None: ax.set_ylim(*ylims)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.tight_layout()
        
        # Save and close
        if saveas: plt.savefig(saveas, bbox_inches='tight')
        if not show: plt.close()
        return ax

    ######################
    ##  Helper Methods  ##
    ######################

    @eqx.filter_jit
    def _sample_y0(self, key, y0):
        y0_samp = jnp.empty([y0.shape[0], self.ncells, y0.shape[2]])
        if y0.shape[1] < self.ncells:
            # Sample with replacement
            for bidx in range(y0.shape[0]):
                key, subkey = jrandom.split(key, 2)
                samp_idxs = jnp.array(
                    jrandom.choice(subkey, y0.shape[1], (self.ncells,), True),
                    dtype=int,
                )
                # y0_samp[bidx,:] = y0[bidx,samp_idxs]
                y0_samp = y0_samp.at[bidx,:].set(y0[bidx,samp_idxs])
        else:
            # Sample without replacement
            for bidx in range(y0.shape[0]):
                key, subkey = jrandom.split(key, 2)
                samp_idxs = jnp.array(
                    jrandom.choice(subkey, y0.shape[1], (self.ncells,), False),
                    dtype=int,
                )
                y0_samp = y0_samp.at[bidx,:].set(y0[bidx,samp_idxs])
        return y0_samp
    

##########################
##  Model Construction  ##
##########################

def make_model(
    key, *, 
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
    dtype=jnp.float32,
):
    """Construct a model and store all hyperparameters.
    
    Args:
        key
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
        dtype
    
    Returns:
        PLNN: model.
        dict: dictionary of hyperparameters.
    """
    model = PLNN(
        key=key,
        ndims=ndims, nparams=nparams, nsigs=nsigs, ncells=ncells,
        signal_type=signal_type, nsigparams=nsigparams,
        sigma_init=sigma_init,
        solver=solver, dt0=dt0, confine=confine, sample_cells=sample_cells,
        infer_metric=infer_metric,
        include_phi_bias=include_phi_bias,
        include_tilt_bias=include_tilt_bias,
        include_metric_bias=include_metric_bias,
        phi_hidden_dims=phi_hidden_dims, 
        phi_hidden_acts=phi_hidden_acts, 
        phi_final_act=phi_final_act,
        phi_layer_normalize=phi_layer_normalize,
        tilt_hidden_dims=tilt_hidden_dims, 
        tilt_hidden_acts=tilt_hidden_acts, 
        tilt_final_act=tilt_final_act,
        tilt_layer_normalize=tilt_layer_normalize,
        metric_hidden_dims=metric_hidden_dims, 
        metric_hidden_acts=metric_hidden_acts, 
        metric_final_act=metric_final_act,
        metric_layer_normalize=metric_layer_normalize,
        dtype=dtype,
    )
    hyperparams = model.get_hyperparameters()
    # Append to dictionary those hyperparams not stored internally.
    hyperparams.update({
        'phi_hidden_dims' : phi_hidden_dims,
        'phi_hidden_acts' : phi_hidden_acts,
        'phi_final_act' : phi_final_act,
        'phi_layer_normalize' : phi_layer_normalize,
        'tilt_hidden_dims' : tilt_hidden_dims,
        'tilt_hidden_acts' : tilt_hidden_acts,
        'tilt_final_act' : tilt_final_act,
        'tilt_layer_normalize' : tilt_layer_normalize,
        'metric_hidden_dims' : metric_hidden_dims,
        'metric_hidden_acts' : metric_hidden_acts,
        'metric_final_act' : metric_final_act,
        'metric_layer_normalize' : metric_layer_normalize,
    })
    return model, hyperparams


############################
##  Model Saving/Loading  ##
############################

def save_model(fname, model, hyperparams):
    """Save a model and hyperparameters to output file.

    Args:
        fname (str): Output file name.
        model (PLNN): Model instance.
        hyperparams (dict): Hyperparameters of the model.
    """
    with open(fname, "wb") as f:
        hyperparam_str = json.dumps(hyperparams)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)  

def load_model(fname)->tuple[PLNN,dict]:
    """Load a model from a binary parameter file.
    
    Args:
        fname (str): Parameter file to load.

    Returns:
        PLNN: Model instance.
    """
    with open(fname, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        model, _ = make_model(key=jrandom.PRNGKey(0), **hyperparams)
        return eqx.tree_deserialise_leaves(f, model), hyperparams


############################
##  Model Initialization  ##
############################

def initialize_model(
    key, model, dtype=jnp.float32, *, 
    init_phi_weights_method='xavier_uniform',
    init_phi_weights_args=[],
    init_phi_bias_method='constant',
    init_phi_bias_args=[0.],
    init_tilt_weights_method='xavier_uniform',
    init_tilt_weights_args=[],
    init_tilt_bias_method='constant',
    init_tilt_bias_args=[0.],
    init_metric_weights_method='xavier_uniform',
    init_metric_weights_args=[],
    init_metric_bias_method='constant',
    init_metric_bias_args=[0.],
):
    if 'xavier_uniform' in [init_phi_bias_method, init_tilt_bias_method, 
                            init_metric_bias_method]:
        raise RuntimeError("Cannot initialize bias using `xavier_uniform`")
    key, key1, key2, key3, key4, key5, key6 = jrandom.split(key, 7)
    is_linear = lambda x: isinstance(x, eqx.nn.Linear)
    
    # Initialize PLNN Weights
    get_weights = lambda m: [
            x.weight 
            for x in jax.tree_util.tree_leaves(m.phi_nn, is_leaf=is_linear) 
            if is_linear(x)
        ]
    init_fn_args = _get_nn_init_args(init_phi_weights_args)
    init_fn_handle = _get_nn_init_func(init_phi_weights_method)
    if init_fn_handle:
        init_fn = init_fn_handle(*init_fn_args)
        weights = get_weights(model)
        new_weights = [
            init_fn(subkey, w.shape, dtype) 
            for w, subkey in zip(weights, jrandom.split(key1, len(weights)))
        ]
        model = eqx.tree_at(get_weights, model, new_weights)

    # Initialize PLNN Bias if applicable
    get_biases = lambda m: [
            x.bias 
            for x in jax.tree_util.tree_leaves(m.phi_nn, is_leaf=is_linear) 
            if is_linear(x) and x.use_bias
        ]
    init_fn_args = _get_nn_init_args(init_phi_bias_args)
    init_fn_handle = _get_nn_init_func(init_phi_bias_method)
    if init_fn_handle and model.include_phi_bias:
        init_fn = init_fn_handle(*init_fn_args)
        biases = get_biases(model)
        new_biases = [
            init_fn(subkey, b.shape, dtype) 
            for b, subkey in zip(biases, jrandom.split(key2, len(biases)))
        ]
        model = eqx.tree_at(get_biases, model, new_biases)

    # Initialize TiltNN Weights
    get_weights = lambda m: [
            x.weight 
            for x in jax.tree_util.tree_leaves(m.tilt_nn, is_leaf=is_linear) 
            if is_linear(x)
        ]
    init_fn_args = _get_nn_init_args(init_tilt_weights_args)
    init_fn_handle = _get_nn_init_func(init_tilt_weights_method)
    if init_fn_handle:
        init_fn = init_fn_handle(*init_fn_args)
        weights = get_weights(model)
        new_weights = [
            init_fn(subkey, w.shape, dtype) 
            for w, subkey in zip(weights, jrandom.split(key3, len(weights)))
        ]
        model = eqx.tree_at(get_weights, model, new_weights)

    # Initialize TiltNN Bias if applicable
    get_biases = lambda m: [
            x.bias 
            for x in jax.tree_util.tree_leaves(m.tilt_nn, is_leaf=is_linear) 
            if is_linear(x) and x.use_bias
        ]
    init_fn_args = _get_nn_init_args(init_tilt_bias_args)
    init_fn_handle = _get_nn_init_func(init_tilt_bias_method)
    if init_fn_handle and model.include_tilt_bias:
        init_fn = init_fn_handle(*init_fn_args)
        biases = get_biases(model)
        new_biases = [
            init_fn(subkey, b.shape, dtype) 
            for b, subkey in zip(biases, jrandom.split(key4, len(biases)))
        ]
        model = eqx.tree_at(get_biases, model, new_biases)

    # Initialize MetricNN Weights
    get_weights = lambda m: [
            x.weight 
            for x in jax.tree_util.tree_leaves(m.metric_nn, is_leaf=is_linear) 
            if is_linear(x)
        ]
    init_fn_args = _get_nn_init_args(init_metric_weights_args)
    init_fn_handle = _get_nn_init_func(init_metric_weights_method)
    if init_fn_handle:
        init_fn = init_fn_handle(*init_fn_args)
        weights = get_weights(model)
        new_weights = [
            init_fn(subkey, w.shape, dtype) 
            for w, subkey in zip(weights, jrandom.split(key5, len(weights)))
        ]
        model = eqx.tree_at(get_weights, model, new_weights)

    # Initialize MetricNN Bias if applicable
    get_biases = lambda m: [
            x.bias 
            for x in jax.tree_util.tree_leaves(m.metric_nn, is_leaf=is_linear) 
            if is_linear(x) and x.use_bias
        ]
    init_fn_args = _get_nn_init_args(init_metric_bias_args)
    init_fn_handle = _get_nn_init_func(init_metric_bias_method)
    if init_fn_handle and model.include_metric_bias:
        init_fn = init_fn_handle(*init_fn_args)
        biases = get_biases(model)
        new_biases = [
            init_fn(subkey, b.shape, dtype) 
            for b, subkey in zip(biases, jrandom.split(key6, len(biases)))
        ]
        model = eqx.tree_at(get_biases, model, new_biases)

    return model


def _get_nn_init_func(init_method):
    if init_method is None:
        init_method = 'none'
    init_method = init_method.lower()
    if init_method not in _INIT_METHOD_KEYS:
        msg = f"Unknown nn layer initialization method: {init_method}"
        raise RuntimeError(msg)
    return _INIT_METHOD_KEYS[init_method]
    

def _get_nn_init_args(init_args):
    if init_args is None:
        return []
    elif isinstance(init_args, list):
        return init_args
    else:
        return [init_args]
    