"""Landscape Potential Model

"""

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, Float  # https://github.com/google/jaxtyping
from diffrax import diffeqsolve, ControlTerm, MultiTerm, ODETerm, SaveAt
from diffrax import Euler, ReversibleHeun, VirtualBrownianTree
import equinox as eqx
from functools import partial

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
    
    ndim: int
    nsig: int
    ncells: int
    nsigparams: int
    phi_nn: eqx.Module
    tilt_nn: eqx.Module
    logsigma: Array
    sigma_init: float
    signal_type: str
    dt0: float
    sample_cells: bool
    include_phi_bias: bool
    include_signal_bias: bool

    def __init__(
        self, 
        key,
        ndim,
        nsig,
        ncells,
        signal_type='jump',
        nsigparams=5,
        sigma_init=1e-2,
        dt0=1e-2,
        hidden_dims=[16, 32, 32, 16],
        hidden_acts='elu',
        final_act='softplus',
        layer_normalize=False,
        include_phi_bias=True,
        include_signal_bias=False,
        dtype=jnp.float32,
        sample_cells=True,
    ):
        """
        """
        super().__init__()
        key, key1, key2 = jax.random.split(key, 3)
        
        self.ndim = ndim
        self.nsig = nsig
        self.ncells = ncells
        self.nsigparams = nsigparams
        self.sigma_init = sigma_init
        self.signal_type = signal_type
        self.dt0 = dt0
        self.sample_cells = sample_cells
        self.include_phi_bias = include_phi_bias
        self.include_signal_bias = include_signal_bias

        # Potential Network hidden layer specifications
        hidden_dims, hidden_acts, final_act = self._check_hidden_layers(
            hidden_dims=hidden_dims, 
            hidden_acts=hidden_acts, 
            final_act=final_act,
        )

        # Potential Neural Network: Maps ndims to a scalar.
        self.phi_nn = self._construct_phi_nn(
            key1, hidden_dims, hidden_acts, final_act, layer_normalize,
            bias=include_phi_bias, dtype=dtype
        )
        
        # Tilt Neural Network: Linear tilt values. Maps nsigs to ndims.
        self.tilt_nn = self._construct_tilt_nn(
            key2, 
            bias=include_signal_bias, dtype=dtype
        )

        # Noise inference or constant
        self.logsigma = jnp.array(jnp.log(sigma_init), dtype=dtype)

    def __call__(
        self, 
        t0: Float[Array, "b"],
        t1: Float[Array, "b"],
        y0: Float[Array, "b ncells ndim"],
        sigparams: Float[Array, "b nsigparams"],
        key: Array,
    ) -> Float[Array, "b ncells ndim"]:
        """Forward call.

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
        key, subkey = jrandom.split(key, 2)
        if self.sample_cells:
            y0 = self._sample_y0(subkey, y0)
        keys = jax.random.split(key, t0.shape[0])
        return fwdvec(t0, t1, y0, sigparams, keys)
    
    ######################
    ##  Getter Methods  ##
    ######################
    
    def get_ncells(self):
        return self.ncells

    def get_sigma(self):
        return jnp.exp(self.logsigma.item())
    
    def get_parameters(self):
        def linear_layers(m):
            return [x for x in m.layers if isinstance(x, eqx.nn.Linear)]
        phi_linlayers  = linear_layers(self.phi_nn)
        tilt_linlayers = linear_layers(self.tilt_nn)
        phi_params = []
        tilt_params = []
        for layer in phi_linlayers:
            phi_params.append(layer.weight)
            if layer.bias is not None:
                phi_params.append(layer.bias)
        for layer in tilt_linlayers:
            tilt_params.append(layer.weight)
            if layer.bias is not None:
                tilt_params.append(layer.bias)
        return phi_params + tilt_params
    
    def get_hyperparameters(self):
        return {
            'ndim' : self.ndim,
            'nsig' : self.nsig,
            'ncells' : self.ncells,
            'nsigparams' : self.nsigparams,
            'sigma_init' : self.sigma_init,
        }
    
    ##########################
    ##  Simulation Methods  ##
    ##########################

    @eqx.filter_jit
    def simulate_forward(
        self,
        t0: Float,
        t1: Float,
        y0: Float[Array, "ncells ndim"],
        sigparams: Float[Array, "5"],
        key: Array,
    )->Float[Array, "ncells ndim"]:
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
        y0: Float[Array, "ndim"],
        sigparams: Float[Array, "5"],
        key: Array,
    ):
        """TODO
        """
        drift = lambda t, y, args: self.eval_f(t, y, sigparams)
        diffusion = lambda t, y, args: self.eval_g(t, y)
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
        y: Float[Array, "ndim"], 
        sig_params: Float[Array, "nsigparams"],
    ) -> Float[Array, "ndim"]:
        """Evaluate drift term. 

        Args:
            t (Scalar)         : Time.
            y (Array)          : State. Shape (d,).
            sig_params (Array) : Signal parameters. Shape (nsigparams,).
        Returns:
            Array of shape (d,).
        """
        gphi = self.eval_grad_phi(t, y)
        gtilt = self.grad_tilt(t, sig_params)
        return -(gphi + gtilt)

    @eqx.filter_jit
    def eval_g(
        self, 
        t: Float, 
        y: Float[Array, "ndim"]
    ) -> Float[Array, "ndim"]:
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
    def eval_phi(
        self, 
        y: Float[Array, "ndim"]
    ) -> Float:
        """Evaluate potential value without tilt.

        Args:
            y (Array) : State. Shape (d,).
        Returns:
            Array of shape (1,).
        """
        return self.phi_nn(y).squeeze(-1)

    @eqx.filter_jit
    def eval_grad_phi(
        self, 
        t: Float, 
        y: Float[Array, "ndim"]
    ) -> Float[Array, "ndim"]:
        """Evaluate gradient of potential without tilt.

        Args:
            t (Scalar) : Time.
            y (Array)  : State. Shape (d,).
        Returns:
            Array of shape (d,).
        """
        return jax.jacrev(self.phi_nn)(y).squeeze(0)
    
    @eqx.filter_jit
    def grad_tilt(
        self, 
        t: Float, 
        sig_params: Float[Array, "nsigparams"]
    ) -> Float[Array, "nsigs"]:
        """Evaluate gradient of linear tilt function. 

        Args:
            t (Scalar) : Time.
            sig_params (Array) : Signal parameters. Shape (nsigparams,).
        Returns:
            Array of shape (nsigs,).
        """
        signal_vals = self.binary_signal_function(t, sig_params)
        return self.tilt_nn(signal_vals)
    
    ####################################
    ##  Vectorized Landscape Methods  ##
    ####################################
    
    @eqx.filter_jit
    def f(
        self, 
        t: Float, 
        y: Float[Array, "ncells ndim"], 
        sig_params: Float[Array, "nsigparams"],
    ) -> Float[Array, "ncells ndim"]:
        """Drift term vectorized across a set of cells.

        Args:
            t (Scalar)         : Time.
            y (Array)          : State. Shape (n,d).
            sig_params (Array) : Signal parameters. Shape (nsigparams,).
        Returns:
            Array of shape (d,).
        """
        return jax.vmap(self.eval_f, (None, 0, None))(t, y, sig_params)

    @eqx.filter_jit
    def g(
        self, 
        t: Float, 
        y: Float[Array, "ncells ndim"]
    ) -> Float[Array, "ncells ndim"]:
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
        y: Float[Array, "ncells ndim"]
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
        y: Float[Array, "ncells ndim"]
    ) -> Float[Array, "ncells ndim"]:
        """Gradient of potential without tilt, vectorized across a set of cells.

        Args:
            t (Scalar) : Time.
            y (Array)  : State. Shape (n,d).
        Returns:
            Array of shape (n,d).
        """
        return jax.vmap(self.eval_grad_phi, (None, 0))(t, y)
        
    ##########################################
    ##  Batch-Vectorized Landscape Methods  ##
    ##########################################

    @eqx.filter_jit
    def f_batched(
        self, 
        t: Float[Array, "b 1"], 
        y: Float[Array, "b ncells ndim"], 
        sig_params: Float[Array, "b nsigparams"],
    ) -> Float[Array, "b ncells ndim"]:
        """Batch-vectorized drift term. 

        Args:
            t (Array)          : Time values. Shape (b,).
            y (Array)          : State vector. Shape (b,n,d).
            sig_params (Array) : Signal parameters. Shape (b,nsigparams,).
        Returns:
            Array of shape (b,n,d).
        """
        return jax.vmap(self.f_vec, 0)(t, y, sig_params)
    
    @eqx.filter_jit
    def phi_batched(
        self, 
        y: Float[Array, "b ncells ndim"]
    ) -> Float[Array, "b ncells 1"]:
        """Batch-vectorized potential value, without tilt. 

        Args:
            y (Array) : State vector. Shape (b,n,d).
        Returns:
            Array of shape (b,n,1).
        """
        return jax.vmap(self.phi_vec, 0)(y)

    @eqx.filter_jit
    def grad_phi_batched(
        self, 
        t: Float[Array, "b 1"], 
        y: Float[Array, "b ncells ndim"]
    ) -> Float[Array, "b ncells ndim"]:
        """Batch-vectorized gradient of potential, without tilt. 

        Args:
            t (Array) : Time values. Shape (b,).
            y (Array) : State vector. Shape (b,n,d).
        Returns:
            Array of shape (b,n,d).
        """
        return jax.vmap(self.grad_phi_vec, 0)(t, y)
        
    ########################
    ##  Signal Functions  ##
    ########################

    @eqx.filter_jit
    def binary_signal_function(self, t, sigparams):
        """TODO
        """
        tcrit = sigparams[...,0]
        p0 = sigparams[...,1:3]
        p1 = sigparams[...,3:5]
        return (t < tcrit) * p0 + (t >= tcrit) * p1

    ##############################
    ##  Initialization Methods  ##
    ##############################

    def _check_hidden_layers(self, hidden_dims, hidden_acts, final_act):
        """Check the model architecture.
        """
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
        layer_list = []
        # Hidden layers
        key, subkey = jrandom.split(key, 2)
        self._add_layer(
            subkey, 
            layer_list, self.ndim, hidden_dims[0], 
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
            layer_list, hidden_dims[-1], 1,  
            activation=final_act, 
            normalization=False,
            bias=bias,
            dtype=dtype,
        )
        return eqx.nn.Sequential(layer_list)
    
    def _construct_tilt_nn(self, key, bias, dtype=jnp.float32):
        layer_list = [eqx.nn.Linear(self.nsig, self.ndim,
                                    use_bias=bias, key=key)]
        return eqx.nn.Sequential(layer_list)

    ######################
    ##  Helper Methods  ##
    ######################

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

def make_model(key, ndim, nsig, ncells, 
               signal_type, nsigparams, sigma_init, 
               hidden_dims, hidden_acts, final_act, layer_normalize, 
               include_phi_bias, include_signal_bias, sample_cells, 
               dt0, dtype=jnp.float32):
    hyperparams = {
        'ndim' : ndim,
        'nsig' : nsig,
        'ncells' : ncells,
        'signal_type' : signal_type,
        'nsigparams' : nsigparams,
        'sigma_init' : sigma_init,
        'hidden_dims' : hidden_dims,
        'hidden_acts' : hidden_acts,
        'final_act' : final_act,
        'layer_normalize' : layer_normalize,
        'include_phi_bias' : include_phi_bias,
        'include_signal_bias' : include_signal_bias,
        'sample_cells' : sample_cells,
        'dt0' : dt0,
        'dtype' : dtype,
    }
    model = PLNN(
        key=key, 
        ndim=ndim,
        nsig=nsig,
        ncells=ncells,
        signal_type=signal_type,
        nsigparams=nsigparams,
        sigma_init=sigma_init,
        hidden_dims=hidden_dims,
        hidden_acts=hidden_acts,
        final_act=final_act,
        layer_normalize=layer_normalize,
        include_phi_bias=include_phi_bias,
        include_signal_bias=include_signal_bias,
        sample_cells=sample_cells,
        dt0=dt0,
        dtype=dtype,
    )
    return model, hyperparams


############################
##  Model Initialization  ##
############################

def initialize_model(
    key,
    model, 
    dtype=jnp.float32,
    init_phi_weights_method='xavier_uniform',
    init_phi_weights_args=[],
    init_phi_bias_method='constant',
    init_phi_bias_args=[0.],
    init_tilt_weights_method='xavier_uniform',
    init_tilt_weights_args=[],
    init_tilt_bias_method='constant',
    init_tilt_bias_args=[0.],
):
    if 'xavier_uniform' in [init_phi_bias_method, init_tilt_bias_method]:
        raise RuntimeError("Cannot initialize bias using `xavier_uniform`")
    key, key1, key2, key3, key4 = jrandom.split(key, 5)
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
    if init_fn_handle and model.include_signal_bias:
        init_fn = init_fn_handle(*init_fn_args)
        biases = get_biases(model)
        new_biases = [
            init_fn(subkey, b.shape, dtype) 
            for b, subkey in zip(biases, jrandom.split(key4, len(biases)))
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
    