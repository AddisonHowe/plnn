"""Abstract Base Class for a Parameterized Landscape.

"""

import json
from abc import abstractmethod
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, Float
import diffrax
from diffrax import diffeqsolve, WeaklyDiagonalControlTerm, MultiTerm, ODETerm
from diffrax import VirtualBrownianTree, SaveAt, SubSaveAt
from diffrax import PIDController, ConstantStepSize
import equinox as eqx

import plnn.pl as pl

MAX_STEPS = 4096*8
RTOL = 1e-5
ATOL = 1e-6
DTMIN = 1e-3
DTMAX = 1.

_ACTIVATION_KEYS = {
    'none' : None,
    'softplus' : jax.nn.softplus,
    'elu' : jax.nn.elu,
    'tanh' : jax.nn.tanh,
    'square' : lambda x: x*x,
}

_SOLVER_KEYS = {
    'euler' : diffrax.Euler, 
    'heun' : diffrax.Heun, 
    'reversible_heun' : diffrax.ReversibleHeun, 
    'ito_milstein' : diffrax.ItoMilstein, 
    'stratonovich_milstein': diffrax.StratonovichMilstein,
}

def _nocall(*args, **kwargs):
    return ConstantStepSize()

_PIDC_KEYS = {
    'euler' : _nocall, 
    'heun' : _nocall, 
    'reversible_heun' : PIDController, 
    'ito_milstein' : _nocall, 
    'stratonovich_milstein': _nocall,
}

def _explicit_initilizer(values):
    counter = 0
    def init_fn(key, shape, dtype):
        nonlocal counter
        v = jnp.empty(shape, dtype=dtype)
        v = v.at[:].set(jnp.array(values[counter], dtype=dtype))
        counter += 1
        return v
    return init_fn

_INIT_METHOD_KEYS = {
    'none' : None,
    'xavier_uniform' : jax.nn.initializers.glorot_uniform,  # no parameters
    'constant' : jax.nn.initializers.constant,  # parameters (1): constant value
    'normal' : jax.nn.initializers.normal,  # parameters (1): std
    'explicit' : _explicit_initilizer, # parameters (1): values
}

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


##############################################################################
#########################  Abstract PLNN Base Class  #########################
##############################################################################

class PLNN(eqx.Module):

    phi_module: eqx.Module     # learnable (abstract)
    tilt_module: eqx.Module    # learnable
    logsigma: Array            # learnable TODO: make Array[Float]
    
    model_type: str  # (abstract)
    ndims: int
    nparams: int
    nsigs: int
    ncells: int
    signal_type: str
    nsigparams: int
    sigma_init: float
    solver: str
    dt0: float
    vbt_tol: float
    confine: bool
    confinement_factor: float
    sample_cells: bool
    include_tilt_bias: bool

    def __init__(
        self, 
        key, *,
        dtype,
        ndims,
        nparams,
        nsigs,
        ncells,
        signal_type,
        nsigparams,
        sigma_init,
        solver,
        dt0,
        vbt_tol,
        confine,
        confinement_factor,
        sample_cells,
        include_tilt_bias,
        tilt_hidden_dims,
        tilt_hidden_acts,
        tilt_final_act,
        tilt_layer_normalize,
    ):
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
        self.vbt_tol = vbt_tol
        self.confine = confine
        self.confinement_factor = confinement_factor
        self.sample_cells = sample_cells
        self.include_tilt_bias = include_tilt_bias

        key, key_tilt = jax.random.split(key, 2)

        # Noise parameter
        self.logsigma = jnp.array(jnp.log(sigma_init), dtype=dtype)

        # Tilt Neural Network: Linear tilt values. Maps nsigs to ndims.
        self.tilt_module = self._construct_tilt_module(
            key_tilt, 
            tilt_hidden_dims,
            tilt_hidden_acts,
            tilt_final_act,
            tilt_layer_normalize,
            bias=include_tilt_bias, 
            dtype=dtype
        )

    #######################
    ##  Forward Methods  ##
    #######################

    def __call__(
        self, 
        t0: Float[Array, "b"],
        t1: Float[Array, "b"],
        y0: Float[Array, "b ncells ndims"],
        sigparams: Float[Array, "b nsigs nsigparams"],
        key: Array,
    ) -> Float[Array, "b ncells ndims"]:
        """Forward call. Acts on batched data.

        Simulate across batches a number of initial conditions between times t0
        and t1, where t0 and t1 are arrays.
        
        Args:
            t0 (Array) : Initial times. Shape (b,).
            t1 (Array) : End times. Shape (b,).
            y0 (Array) : Initial conditions. Shape (b,n,d).
            sigparams (Array) : Signal parameters. Shape (b,nsigs,nsigparams).
            key (Array) : PRNGKey.

        Returns:
            (Array) Final states, across batches. Shape (b,n,d).
        """
        # Parse the inputs
        fwdvec = jax.vmap(self.simulate_ensemble, 0)
        key, sample_key = jrandom.split(key, 2)
        if self.sample_cells and self.ncells > 1:
            y0 = self._sample_y0(sample_key, y0)
        batch_keys = jax.random.split(key, t0.shape[0])
        return fwdvec(t0, t1, y0, sigparams, batch_keys)

    ######################
    ##  Getter Methods  ##
    ######################
    
    def get_ncells(self) -> int:
        return self.ncells

    def get_sigma(self) -> Float[Array, "1"]:
        return jnp.exp(self.logsigma.item())
    
    def get_parameters(self) -> dict:
        """Return dictionary of learnable model parameters.

        The returned dictionary contains the following strings as keys: 
            tilt.w, tilt.b, sigma
        Each key maps to a list of jnp.array objects.
        
        Returns:
            dict: Dictionary containing learnable parameters.
                
        """
        tilt_linlayers = self._get_linear_module_layers(self.tilt_module)
        d = {
            'tilt.w' : [l.weight for l in tilt_linlayers],
            'tilt.b' : [l.bias for l in tilt_linlayers],
            'sigma'  : self.get_sigma(),
        }
        return d
    
    def get_hyperparameters(self) -> dict:
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
            'vbt_tol': self.vbt_tol,
            'confine' : self.confine,
            'confinement_factor' : self.confinement_factor,
            'sample_cells' : self.sample_cells,
            'include_tilt_bias' : self.include_tilt_bias,
        }
        
    def get_linear_layer_parameters(self) -> list[Array]:
        """Return a list of learnable parameters from linear layers.
        
        Returns:
            list[Array] : List of linear layer learnable parameter arrays.
        """
        tilt_linlayers = self._get_linear_module_layers(self.tilt_module)
        tilt_params = []
        for layer in tilt_linlayers:
            tilt_params.append(layer.weight)
            if layer.bias is not None:
                tilt_params.append(layer.bias)
        return tilt_params

    def get_info_string(self, usetex=True) -> str:
        """Latex format string with model information."""
        if usetex:
            return f"\\noindent PLNN: \\texttt{{{self.model_type}}}"
        else:
            return f"PLNN<{self.model_type}>"

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
        return self.eval_f(t, y, sigparams)
    
    def diffusion(
            self,
            t: Float, 
            y: Float[Array, "ndims"], 
            args=None
    ) -> Float[Array, "ndims"]:
        """Diffusion term used in `simulate_path` method.

        Args:
            t (Scalar) : Time.
            y (Array)  : State. Shape (d,).

        Returns:
            Array : Diffusion term. Shape (d,).
        """
        return self.eval_g(t, y)
    
    def simulate_path(
        self,
        t0: Float,
        t1: Float,
        y0: Float[Array, "ndims"],
        sigparams: Float[Array, "nsigs nsigparams"],
        key: Array,
    )->Float[Array, "ndims"]:
        """Evolve a single cell forward in time and return the final state.
        
        Args:
            t0 (Array) : Initial time. Shape (1,).
            t1 (Array) : End time. Shape (1,).
            y0 (Array) : Initial condition. Shape (d,).
            sigparams (Array) : Signal parameters. Shape (nsigs,nsigparams).
            key (Array) : PRNGKey.

        Returns:
            Array : Final state. Shape (d,).
        """
        brownian_motion = VirtualBrownianTree(
            t0, 
            t1, 
            tol=self.vbt_tol, 
            shape=(len(y0),), 
            key=key
        )
        terms = MultiTerm(
            ODETerm(self.drift), 
            WeaklyDiagonalControlTerm(self.diffusion, brownian_motion)
        )
        solver = _SOLVER_KEYS[self.solver]()
        pidc = _PIDC_KEYS[self.solver](
            rtol=RTOL,
            atol=ATOL,
            dtmin=DTMIN,
            dtmax=DTMAX,
        )
        saveat = SaveAt(t1=True)
        sol = diffeqsolve(
            terms, solver, 
            t0, t1, dt0=self.dt0, 
            y0=y0, 
            saveat=saveat,
            args=sigparams,
            stepsize_controller=pidc,
            max_steps=MAX_STEPS,
        )
        return sol.ys

    def simulate_ensemble(
        self,
        t0: Float,
        t1: Float,
        y0: Float[Array, "ncells ndims"],
        sigparams: Float[Array, "nsigs nsigparams"],
        key: Array,
    )->Float[Array, "ncells ndims"]:
        """Evolve an ensemble forward in time and return the final state.
        
        Args:
            t0 (Array) : Initial time. Shape (1,).
            t1 (Array) : End time. Shape (1,).
            y0 (Array) : Initial condition. Shape (n,d).
            sigparams (Array) : Signal parameters. Shape (nsigs,nsigparams).
            key (Array) : PRNGKey.

        Returns:
            Array : Final state. Shape (n,d).
        """
        subkeys = jrandom.split(key, len(y0))
        vecsim = jax.vmap(self.simulate_path, (None, None, 0, None, 0))
        return vecsim(t0, t1, y0, sigparams, subkeys).squeeze(1)
    
    def simulate_path_with_saves(
        self,
        t0: Float,
        t1: Float,
        y0: Float[Array, "ndims"],
        sigparams: Float[Array, "nsigs nsigparams"],
        saveat: SaveAt,
        key: Array,
    )->Float[Array, "? ndims"]:
        """Evolve a single cell forward in time and return all evaluated states.
        
        Args:
            t0 (Array) : Initial time. Shape (1,).
            t1 (Array) : End time. Shape (1,).
            y0 (Array) : Initial condition. Shape (d,).
            sigparams (Array) : Signal parameters. Shape (nsigs,nsigparams).
            saveat (diffrax.SaveAt) : Times at which to save.
            key (Array) : PRNGKey.

        Returns:
            Array : Final state. Shape (?,d).
        """
        brownian_motion = VirtualBrownianTree(
            t0, 
            t1, 
            tol=self.vbt_tol, 
            shape=(len(y0),), 
            key=key
        )
        terms = MultiTerm(
            ODETerm(self.drift), 
            WeaklyDiagonalControlTerm(self.diffusion, brownian_motion)
        )
        solver = _SOLVER_KEYS[self.solver]()
        pidc = _PIDC_KEYS[self.solver](
            rtol=RTOL,
            atol=ATOL,
            dtmin=DTMIN,
            dtmax=DTMAX,
        )
        sol = diffeqsolve(
            terms, solver, 
            t0, t1, dt0=self.dt0, 
            y0=y0, 
            saveat=saveat,
            args=sigparams,
            stepsize_controller=pidc, 
            max_steps=MAX_STEPS,
        )
        return sol.ts, sol.ys
    
    def simulate_ensemble_with_saves(
        self,
        t0: Float,
        t1: Float,
        y0: Float[Array, "ncells ndims"],
        sigparams: Float[Array, "nsigs nsigparams"],
        saveat: SaveAt,
        key: Array,
    )->Float[Array, "? ncells ndims"]:
        """Evolve an ensemble forward in time and return all evaluated states.
        
        Args:
            t0 (Array) : Initial time. Shape (1,).
            t1 (Array) : End time. Shape (1,).
            y0 (Array) : Initial condition. Shape (n,d).
            sigparams (Array) : Signal parameters. Shape (nsigs,nsigparams).
            saveat (diffrax.SaveAt) : Times at which to save.
            key (Array) : PRNGKey.

        Returns:
            Array : Final state. Shape (?,n,d).
        """
        subkeys = jrandom.split(key, len(y0))
        vecsim = jax.vmap(self.simulate_path_with_saves, (None, None, 0, None, None, 0))
        return vecsim(t0, t1, y0, sigparams, saveat, subkeys)
    
    def burnin_path(
            self,
            tburn: Float,
            y0: Float[Array, "ndims"],
            sigparams: Float[Array, "nsigs nsigparams"],
            key: Array,
    )->Float[Array, "ndims"]:
        drift = lambda t, y, args: self.drift(0., y, sigparams)
        diffusion = lambda t, y, args: self.diffusion(0., y)
        brownian_motion = VirtualBrownianTree(
            0., 
            tburn, 
            tol=self.vbt_tol, 
            shape=(len(y0),), 
            key=key
        )
        terms = MultiTerm(
            ODETerm(drift), 
            WeaklyDiagonalControlTerm(diffusion, brownian_motion)
        )
        solver = _SOLVER_KEYS[self.solver]()
        saveat = SaveAt(t1=True)
        pidc = _PIDC_KEYS[self.solver](
            rtol=RTOL,
            atol=ATOL,
            dtmin=DTMIN,
            dtmax=DTMAX,
        )
        sol = diffeqsolve(
            terms, solver, 
            0., tburn, dt0=self.dt0, 
            y0=y0, 
            saveat=saveat,
            stepsize_controller=pidc, 
            max_steps=MAX_STEPS,
        )
        return sol.ys
    
    def burnin_ensemble(
            self,
            tburn: Float,
            y0: Float[Array, "ncells ndims"],
            sigparams: Float[Array, "nsigs nsigparams"],
            key: Array,
    )->Float[Array, "ncells ndims"]:
        subkeys = jrandom.split(key, len(y0))
        vecsim = jax.vmap(self.burnin_path, (None, 0, None, 0))
        return vecsim(tburn, y0, sigparams, subkeys).squeeze(1)
    
    def run_landscape_simulation(
            self,
            x0,
            tfin,
            dt_save,
            sigparams,
            key,
            burnin=1e-2,
    ):
        """TODO: Documentation
        """
        if isinstance(dt_save, (float, int)):
            dt_save = [dt_save]

        subsaves_list = []
        for dt_save_val in dt_save:
            ts_save = jnp.linspace(0, tfin, 1 + int(tfin / dt_save_val))
            subsaves_list.append(
                SubSaveAt(ts=ts_save, fn=lambda t, y, args: y)
            )
        saveat = SaveAt(subs=subsaves_list)

        key, subkey1, subkey2 = jax.random.split(key, 3)
        
        x0 = self.burnin_ensemble(burnin, x0, sigparams, subkey1)

        ts, ys = self.simulate_ensemble_with_saves(
            0, tfin, x0, sigparams, saveat, subkey2
        )
        print(len(ys))
        print(ys[0].shape)
        print(ys[0])

        ts = [tt[0] for tt in ts]
        sigs = [
            jax.vmap(self.compute_signal, (0, None))(tt, sigparams) 
            for tt in ts
        ]

        ps = [
            jax.vmap(self.eval_tilt_params, (0, None))(tt, sigparams) 
            for tt in ts
        ]

        ys = [y.transpose([1, 0, 2]) for y in ys]

        return ts, ys, sigs, ps
    
    ##############################
    ##  Core Landscape Methods  ##
    ##############################

    def eval_f(
        self, 
        t: Float, 
        y: Float[Array, "ndims"], 
        sigparams: Float[Array, "nsigs nsigparams"],
    ) -> Float[Array, "ndims"]:
        """Evaluate drift term. 

        Args:
            t (Scalar)        : Time.
            y (Array)         : State. Shape (d,).
            sigparams (Array) : Signal parameters. Shape (nsigs, nsigparams).
        Returns:
            Array of shape (d,).
        """
        gphi = self.eval_grad_phi(t, y)
        gtilt = self.grad_tilt(t, sigparams)
        return -(gphi + gtilt)

    def eval_g(
        self, 
        t: Float, 
        y: Float[Array, "ndims"]
    ) -> Float[Array, "ndims"]:
        """Evaluate diffusion term. 
        
        The d-dimensional vector returned corresponds to a square, diagonal
        matrix, the diffusion term in the governing SDE. We therefore are 
        implicitly assuming a Weakly Diagonal form for the diffusion, in that
        d independent Wiener processes govern each dimension independently, but 
        it is possible that the noise in either dimension can depend on the
        particle's full state.

        Args:
            t (Scalar) : Time.
            y (Array)  : State. Shape (d,).
        Returns:
            Array of shape (d,).
        """
        return jnp.exp(self.logsigma) * jnp.ones(y.shape)
    
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
            return self.confinement_factor * jnp.sum(jnp.power(y, 4))
        return 0.

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
            return 4. * self.confinement_factor * jnp.power(y, 3)
        return y * 0.

    @abstractmethod
    def eval_phi(
        self, 
        y: Float[Array, "ndims"]
    ) -> Float:
        """(Abstract Method) Evaluate potential value without tilt.

        Args:
            y (Array) : State. Shape (d,).
        Returns:
            Array of shape (1,).
        """
        raise NotImplementedError()

    @abstractmethod
    def eval_grad_phi(
        self, 
        t: Float, 
        y: Float[Array, "ndims"]
    ) -> Float[Array, "ndims"]:
        """(Abstract Method) Evaluate gradient of potential without tilt.

        Args:
            t (Scalar) : Time.
            y (Array)  : State. Shape (d,).
        Returns:
            Array of shape (d,).
        """
        raise NotImplementedError()
    
    def grad_tilt(
        self, 
        t: Float, 
        sigparams: Float[Array, "nsigs nsigparams"]
    ) -> Float[Array, "ndims"]:
        """Evaluate gradient of linear tilt function.
        
        Equal to the tilt vector tau, also given by method eval_tilt_params.

        Args:
            t (Scalar) : Time.
            sigparams (Array) : Signal parameters. Shape (nsigs, nsigparams).
        Returns:
            Array of shape (ndims,).
        """
        signal_vals = self.compute_signal(t, sigparams)
        return self.tilt_module(signal_vals)
    
    def eval_tilted_phi(
        self, 
        t: Float, 
        y: Float[Array, "ndims"], 
        sigparams: Float[Array, "nsigs nsigparams"],
    ) -> Float:
        """Evaluate tilted landscape level. 

        Args:
            t (Scalar)        : Time.
            y (Array)         : State. Shape (d,).
            sigparams (Array) : Signal parameters. Shape (nsigs, nsigparams).
        Returns:
            Array of shape (1,).
        """
        phi = self.eval_phi(y)
        tau = self.grad_tilt(t, sigparams)
        return phi + jnp.dot(tau, y)
    
    def eval_tilted_grad_phi(
        self, 
        t: Float, 
        y: Float[Array, "ndims"], 
        sigparams: Float[Array, "nsigs nsigparams"],
    ) -> Float[Array, "ndims"]:
        """Evaluate tilted gradient. 

        Args:
            t (Scalar)        : Time.
            y (Array)         : State. Shape (d,).
            sigparams (Array) : Signal parameters. Shape (nsigs, nsigparams).
        Returns:
            Array of shape (d,).
        """
        grad_phi = self.eval_grad_phi(y)
        tau = self.grad_tilt(t, sigparams)
        return grad_phi + tau
    
    def eval_jacobian(
        self, 
        t: Float, 
        y: Float[Array, "ndims"], 
        sigparams: Float[Array, "nsigs nsigparams"],
    ) -> Float[Array, "ndims ndims"]:
        """Evaluate drift term. 

        Args:
            t (Scalar)        : Time.
            y (Array)         : State. Shape (d,).
            sigparams (Array) : Signal parameters. Shape (nsigs, nsigparams).
        Returns:
            Array of shape (d,d).
        """
        return jax.jacfwd(self.eval_f, 1)(t, y, sigparams)
    
    ####################################
    ##  Vectorized Landscape Methods  ##
    ####################################
    
    def f(
        self, 
        t: Float, 
        y: Float[Array, "ncells ndims"], 
        sigparams: Float[Array, "nsigs nsigparams"],
    ) -> Float[Array, "ncells ndims"]:
        """Drift term vectorized across an ensemble of cells.

        Args:
            t (Scalar)         : Time.
            y (Array)          : State. Shape (n,d).
            sigparams (Array) : Signal parameters. Shape (nsigs, nsigparams).
        Returns:
            Array of shape (d,).
        """
        return jax.vmap(self.eval_f, (None, 0, None))(t, y, sigparams)

    def g(
        self, 
        t: Float, 
        y: Float[Array, "ncells ndims"]
    ) -> Float[Array, "ncells ndims"]:
        """Diffusion term vectorized across an ensemble of cells. 

        Args:
            t (Scalar) : Time.
            y (Array)  : State. Shape (n,d).
        Returns:
            Array of shape (n,d).
        """
        return jax.vmap(self.eval_g, (None, 0))(t, y)
    
    def phi(
        self, 
        y: Float[Array, "ncells ndims"]
    ) -> Float[Array, "ncells"]:
        """Potential value without tilt, vectorized across an ensemble of cells.

        Args:
            y (Array) : State. Shape (n,d).
        Returns:
            Array of shape (n,1).
        """
        return jax.vmap(self.eval_phi, 0)(y)

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
    
    def tilted_grad_phi(
        self, 
        t: Float, 
        y: Float[Array, "ncells ndims"],
        sigparams: Float[Array, "nsigs nsigparams"]
    ) -> Float[Array, "ncells ndims"]:
        """Tilted landscape level. 

        Args:
            t (Scalar) : Time.
            y (Array)  : State. Shape (n,d).
            sigparams (Array) : Signal parameters. Shape (nsigs, nsigparams).
        Returns:
            Array of shape (n,d).
        """
        return jax.vmap(self.eval_tilted_grad_phi, (None, 0, None))(t, y, sigparams)
    
    def jacobian(
        self, 
        t: Float, 
        y: Float[Array, "ncells ndims"],
        sigparams: Float[Array, "nsigs nsigparams"]
    ) -> Float[Array, "ncells ndims ndims"]:
        """Jacobian of field f, vectorized across a set of cells.

        Args:
            t (Scalar) : Time.
            y (Array)  : State. Shape (n,d).
            sigparams (Array) : Signal parameters. Shape (nsigs, nsigparams).
        Returns:
            Array of shape (n,d,d).
        """
        return jax.vmap(self.eval_jacobian, (None, 0, None))(t, y, sigparams)
    
    #######################################
    ##  Convenience Landscape Functions  ##
    #######################################

    def eval_phi_with_signal(
            self, 
            t: Float, 
            y: Float[Array, "ndims"], 
            signal: Float[Array, "nsigs"]
    ) -> Float:
        """Convenience function. Landscape level with tilt based on signal.

        Acts on individual particle. TODO: Test.

        Args:
            t (Scalar)     : Time.
            y (Array)      : State. Shape (d,).
            signal (Array) : Signal parameters. Shape (nsigs,).
        Returns:
            Array of shape (1,).
        """
        phi = self.eval_phi(y)
        tau = self.tilt_module(signal)
        return phi + jnp.dot(tau, y)
    
    def eval_phi_with_tilts(
            self, 
            t: Float, 
            y: Float[Array, "ndims"], 
            tilts: Float[Array, "ndims"]
    ) -> Float:
        """Convenience function. Landscape level with tilt given directly.

        Acts on individual particle. TODO: Test.

        Args:
            t (Scalar)    : Time.
            y (Array)     : State. Shape (d,).
            tilts (Array) : Tilt parameters. Shape (nsigs,).
        Returns:
            Array of shape (1,).
        """
        return self.eval_phi(y) + jnp.dot(tilts, y)

    def phi_with_signal(
            self,
            t: Float, 
            y: Float[Array, "ncells ndims"],
            signal: Float[Array, "nsigs"]
    ) -> Float[Array, "ncells"]:
        """Convenience function. Landscape level with tilt based on signal.

        Vectorized across a number of particles. TODO: Test.

        Args:
            t (Scalar)     : Time.
            y (Array)      : State. Shape (n,d).
            signal (Array) : Signal values. Shape (nsigs,).
        Returns:
            Array of shape (n,1).
        """
        return jax.vmap(self.eval_phi_with_signal, (None, 0, None))(t, y, signal)
    
    def phi_with_tilts(
            self,
            t: Float, 
            y: Float[Array, "ncells ndims"],
            tilts: Float[Array, "ndims"]
    ) -> Float[Array, "ncells"]:
        """Convenience function. Landscape level with tilt given directly.

        Vectorized across a number of particles. TODO: Test.

        Args:
            t (Scalar)    : Time.
            y (Array)     : State. Shape (n,d).
            tilts (Array) : Tilt values. Shape (ndims,).
        Returns:
            Array of shape (n,1).
        """
        return jax.vmap(self.eval_phi_with_tilts, (None, 0, None))(t, y, tilts)
    
    def eval_grad_phi_with_signal(
            self, 
            t: Float, 
            y: Float[Array, "ndims"], 
            signal: Float[Array, "nsigs"]
    ) -> Float[Array, "ndims"]:
        """Convenience function. Gradient with tilt based on signal.

        Acts on individual particle. TODO: Test.

        Args:
            t (Scalar)     : Time.
            y (Array)      : State. Shape (d,).
            signal (Array) : Signal parameters. Shape (nsigs,).
        Returns:
            Array of shape (d,).
        """
        grad_phi = self.eval_grad_phi(t, y)
        tau = self.tilt_module(signal)
        return grad_phi + tau
    
    def eval_grad_phi_with_tilts(
            self, 
            t: Float, 
            y: Float[Array, "ndims"], 
            tilts: Float[Array, "ndims"]
    ) -> Float[Array, "ndims"]:
        """Convenience function. Gradient with tilt given directly.

        Acts on individual particle. TODO: Test.

        Args:
            t (Scalar)    : Time.
            y (Array)     : State. Shape (d,).
            tilts (Array) : Tilt parameters. Shape (nsigs,).
        Returns:
            Array of shape (d,).
        """
        return self.eval_grad_phi(t, y) + tilts

    def grad_phi_with_signal(
            self,
            t: Float, 
            y: Float[Array, "ncells ndims"],
            signal: Float[Array, "nsigs"]
    ) -> Float[Array, "ncells ndims"]:
        """Convenience function. Gradient with tilt based on signal.

        Vectorized across a number of particles. TODO: Test.

        Args:
            t (Scalar)     : Time.
            y (Array)      : State. Shape (n,d).
            signal (Array) : Signal values. Shape (nsigs,).
        Returns:
            Array of shape (n,d).
        """
        return jax.vmap(self.eval_grad_phi_with_signal, (None, 0, None))(t, y, signal)
    
    def grad_phi_with_tilts(
            self,
            t: Float, 
            y: Float[Array, "ncells ndims"],
            tilts: Float[Array, "ndims"]
    ) -> Float[Array, "ncells ndims"]:
        """Convenience function. Gradient with tilt given directly.

        Vectorized across a number of particles. TODO: Test.

        Args:
            t (Scalar)    : Time.
            y (Array)     : State. Shape (n,d).
            tilts (Array) : Tilt values. Shape (ndims,).
        Returns:
            Array of shape (n,d).
        """
        return jax.vmap(self.eval_grad_phi_with_tilts, (None, 0, None))(t, y, tilts)
        
    ########################
    ##  Signal Functions  ##
    ########################

    def eval_tilt_params(
            self,
            t: Float,
            sigparams: Float[Array, "nsigs nsigparams"]
    ) -> Float[Array, "ndims"]:
        """Evaluates the tilt vector tau at time t.

        Args:
            t (Scalar) : Time.
            sigparams (Array) : Signal parameters. Shape (nparams, nsigparams)
        Returns:
            Array of shape (ndims,)
        """
        signal_vals = self.compute_signal(t, sigparams)
        return self.tilt_module(signal_vals)

    def compute_signal(
            self,
            t: Float,
            sigparams: Float[Array, "nsigs nsigparams"]
    ) -> Float[Array, "nsigs"]:
        """Evaluates the signal vector at time t.

        Args:
            t (Scalar) : Time.
            sigparams (Array) : Signal parameters. Shape (nparams, nsigparams)
        Returns:
            Array of shape (nsigs,)
        """
        if self.signal_type in ('jump', 'step', 'binary'):
            return self.binary_signal_function(t, sigparams)
        elif self.signal_type == "sigmoid":
            return self.sigmoid_signal_function(t, sigparams)
        else:
            raise RuntimeError()
            
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
        return (t < tcrit) * p0 + (t >= tcrit) * p1
    
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
        return p0 + 0.5*(p1 - p0) * (1 + jnp.tanh(r * (t - tcrit)))
    
    ############################
    ##  Model Saving/Loading  ##
    ############################

    def save(self, fname, hyperparams):
        """Save model and hyperparameters to output file.

        Args:
            fname (str): Output file name.
            hyperparams (dict): Hyperparameters of the model.
        """
        with open(fname, "wb") as f:
            hyperparam_str = json.dumps(hyperparams)
            f.write((hyperparam_str + "\n").encode())
            eqx.tree_serialise_leaves(f, self)

    @staticmethod
    @abstractmethod
    def load(fname:str, dtype=jnp.float32) -> tuple['PLNN', dict]:
        """(Abstract Method) Load a model from a binary parameter file.
        
        Args:
            fname (str): Parameter file to load.
            dtype : Datatype. Either jnp.float32 or jnp.float64.

        Returns:
            PLNN: Model instance.
            dict: Model hyperparameters.
        """
        raise NotImplementedError()

    #############################
    ##  Initialization Method  ##
    #############################

    def initialize(
            self, 
            key, dtype=jnp.float32, *,
            init_tilt_weights_method='xavier_uniform',
            init_tilt_weights_args=[],
            init_tilt_bias_method='constant',
            init_tilt_bias_args=[0.],
    ) -> 'PLNN':
        """Initializes the tilt module of a PLNN model instance.

        Args:
            key (_type_): TODO
            dtype (_type_, optional): TODO. Defaults to jnp.float32.
            init_tilt_weights_method (str, optional): TODO. Defaults to 'xavier_uniform'.
            init_tilt_weights_args (list, optional): TODO. Defaults to [].
            init_tilt_bias_method (str, optional): TODO. Defaults to 'constant'.
            init_tilt_bias_args (list, optional): TODO. Defaults to [0.].

        Returns:
            PLNN: Model with initialized weights and biases of the tilt module.
        """
        # Initialize tilt module
        return self._initialize_linear_module(
            key, dtype, self, 'tilt_module', 
            init_weights_method=init_tilt_weights_method,
            init_weights_args=init_tilt_weights_args,
            init_biases_method=init_tilt_bias_method,
            init_biases_args=init_tilt_bias_args,
            include_biases=self.include_tilt_bias,
        )
    
    ###################################
    ##  Construction Helper Methods  ##
    ###################################

    @staticmethod
    def _initialize_linear_module(
            key,
            dtype,
            model,
            module_attribute_name,
            init_weights_method,
            init_weights_args,
            init_biases_method,
            init_biases_args,
            include_biases,
    ) -> 'PLNN':
        """Initialize the weights and biases of a specific model submodule.

        Args:
            key (PRNGKey) : TODO
            dtype () : TODO
            model (PLNN) : TODO
            module_attribute_name (str) : TODO
            init_weights_method (str) : TODO
            init_weights_args () : TODO
            init_biases_method (str) : TODO
            init_biases_args () : TODO
            include_biases (bool) : TODO
        
        Returns:
            (PLNN) Model instance with initialized weights and biases.
        """

        if init_biases_method == 'xavier_uniform':
            msg = "Cannot initialize bias using method: xavier_uniform"
            raise RuntimeError(msg)

        key, key_w, key_b = jrandom.split(key, 3)
        is_linear = lambda x: isinstance(x, eqx.nn.Linear)

        # Initialize weights
        get_weights = lambda m: [
                x.weight for x in jax.tree_util.tree_leaves(
                    getattr(m, module_attribute_name), is_leaf=is_linear
                ) if is_linear(x)
            ]
        init_fn_handle = _get_nn_init_func(init_weights_method)
        init_fn_args = _get_nn_init_args(init_weights_args)
        if init_fn_handle:
            init_fn = init_fn_handle(*init_fn_args)
            weights = get_weights(model)
            old_shapes = [None if w is None else w.shape for w in weights]
            new_weights = [
                init_fn(subkey, w.shape, dtype) for w, subkey 
                in zip(weights, jrandom.split(key_w, len(weights)))
            ]
            new_shapes = [None if w is None else w.shape for w in new_weights]
            if old_shapes != new_shapes:
                msg = f"Weights changed shape during initialization \
                    of module {module_attribute_name}."
                msg += f"\nOld:\n{old_shapes}\nNew:\n{new_shapes}"
                raise RuntimeError(msg)
            model = eqx.tree_at(get_weights, model, new_weights)

        # Initialize biases if specified
        if not include_biases:
            return model
        
        get_biases = lambda m: [
                x.bias 
                for x in jax.tree_util.tree_leaves(
                    getattr(m, module_attribute_name), is_leaf=is_linear) 
                if is_linear(x) and x.use_bias
            ]
        init_fn_handle = _get_nn_init_func(init_biases_method)
        init_fn_args = _get_nn_init_args(init_biases_args)
        if init_fn_handle:
            init_fn = init_fn_handle(*init_fn_args)
            biases = get_biases(model)
            old_shapes = [None if b is None else b.shape for b in biases]
            new_biases = [
                # init_fn(subkey, b.shape, dtype) for b, subkey 
                None if b is None else init_fn(subkey, b.shape, dtype) for b, subkey 
                in zip(biases, jrandom.split(key_b, len(biases)))
            ]
            new_shapes = [None if b is None else b.shape for b in new_biases]
            if old_shapes != new_shapes:
                msg = f"Biases changed shape during initialization."
                msg += f"\nOld:\n{old_shapes}\nNew:\n{new_shapes}"
                raise RuntimeError(msg)
            model = eqx.tree_at(get_biases, model, new_biases)
        
        return model

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
            
    def _construct_tilt_module(self, key, hidden_dims, hidden_acts, final_act, 
                          layer_normalize, bias=False, dtype=jnp.float32):
        return self._construct_ffn(
            key, self.nsigs, self.ndims, 
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

    def plot_phi(
            self, 
            tilt=None,
            signal=None,
            sigparams=None,
            eval_time=None,
            r=4, 
            res=50, 
            plot3d=False, 
            **kwargs
    ):
        """Plot the scalar function phi.
        
        Args:
            r (int) : 
            res (int) :
            plot3d (bool) :
            normalize (bool) :
            lognormalize (bool) :
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
        return pl.plot_phi(
            self,
            tilt=tilt,
            signal=signal,
            sigparams=sigparams,
            eval_time=eval_time,
            r=r,
            res=res,
            plot3d=plot3d, 
            **kwargs
        )
    
    def plot_f(
            self, 
            tilt=None,
            signal=None,
            sigparams=None,
            eval_time=None,
            r=4, 
            res=50, 
            **kwargs):
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
        return pl.plot_f(
            self, 
            tilt=tilt,
            signal=signal,
            sigparams=sigparams,
            eval_time=eval_time,
            r=r, 
            res=res, 
            **kwargs
        )

    ######################
    ##  Helper Methods  ##
    ######################

    @staticmethod
    def _get_linear_module_layers(m):
        return [x for x in m.layers if isinstance(x, eqx.nn.Linear)]

    def _sample_y0(self, key, y0):
        nbatches, _, dim = y0.shape
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
    