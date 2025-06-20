import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import equinox as eqx
# import torch

class Simulator:

    def __init__(self, f, signal_func, param_func, noise_func, metric=None) -> None:
        """
        Args:
            f (callable): deterministic portion of dynamics. Args: t, x, p.
            signal_func (callable): signal function. Arg: t.
            param_func (callable): parameter function. Arg: <signal>.
            noise_func (callable): stochastic portion of dynamics. Args: t, x.
        """
        self.f = f
        self.signal_func = signal_func
        self.param_func = param_func
        self.noise_func = noise_func
        self.metric = metric

    def simulate(
            self, 
            ncells, 
            x0, 
            tfin, 
            dt=1e-2, 
            burnin=0, 
            dt_save=None, 
            rng=np.random.default_rng(),
            param_args=None,
            burnin_signal=None,
    ):
        """Run a simulation.

        Args:
            ncells (int) : Number of particles to simulate.
            x0 (tuple[float]) : Initial state.
            tfin (float) : Simulation end time.
            dt (float) : Simulation internal time step. Must be a divisor of 
                the end time. Default 1e-2.
            burnin (int) : Number of burnin steps to take. Default 0.
            dt_save (float) : Intervals of simulation time at which to save.
                Must be a multiple of the step size dt, but need not divide the
                simulation end time.
            rng (Generator) : Random number generator.
        Returns:
            ts_save (ndarray) : Saved timepoints. Shape (nsaves,).
            xs_save (ndarray) : Saved states. Shape (nsaves, ndims).
            sig_save (ndarray) : Saved signal values. Shape (nsaves, nsignals).
            ps_save (ndarray) : Saved parameter values. Shape (nsaves, nparams).
        """
        # Simulation save points: Save every `saverate` steps
        t0 = 0.
        ts_save, saverate = get_ts_save(tfin, dt, dt_save)
        nsaves = len(ts_save)

        # Simulation timesteps
        ts = np.linspace(0., tfin, 1 + int(tfin / dt))
        
        # Initialize all cells at given state `x0`
        x0 = jnp.array(x0)
        dim = x0.shape[-1]
        xs_save = np.zeros([nsaves, ncells, dim])
        xs_save[0] = x0
        
        # Initialize signals and parameters for burnin phase
        sig0 = self.signal_func(t0)
        if burnin_signal is not None:
            sig0[:] = burnin_signal            
        p0 = self.param_func(0., sig0, param_args)
        sig_shape = sig0.shape
        ps_shape = p0.shape
        sig_save = np.zeros([nsaves, *sig_shape])
        ps_save = np.zeros([nsaves, *ps_shape])
        
        # Initialize noise array
        dw = np.empty(xs_save[0].shape)

        # Initialize state, signal, and parameter arrays
        x = xs_save[0].copy()
        
        @eqx.filter_jit
        def stepper(t, x, dt, dw, include_metric):
            sig = self.signal_func(t)
            p = self.param_func(t, sig, param_args)
            term1 = dt * self.f(t, x, p)
            term2 = dw * self.noise_func(t, x)
            if include_metric:
                g = self.metric(t, x)
                xnew = x + jnp.einsum('ijk,ik->ij', g, term1 + term2)
            else:
                xnew = x + term1 + term2
            return xnew, sig, p
        
        # Burnin steps: Update only x. Signal and parameters are fixed.
        dws = np.sqrt(dt) * rng.standard_normal([burnin, *x.shape])
        for i in range(burnin):
            # dw = np.sqrt(dt) * rng.standard_normal(x.shape)
            dw = dws[i]
            x, _, _ = stepper(t0, x, dt, dw, self.metric is not None)

        # Reinitialize signals and parameters for main steps
        sig0 = self.signal_func(t0)
        p0 = self.param_func(t0, sig0, param_args)

        # Euler steps
        xs_save[0] = x
        sig_save[0] = sig0
        ps_save[0] = p0
        save_counter = 1
        dws = np.sqrt(dt) * rng.standard_normal([len(ts), *x.shape])
        for i, t in zip(range(1, len(ts)), ts[0:-1]):
            dw = dws[i]
            x, sig, p = stepper(t, x, dt, dw, self.metric is not None)
            if i % saverate == 0:
                xs_save[save_counter] = x
                sig_save[save_counter] = sig
                ps_save[save_counter] = p
                save_counter += 1
        
        return ts_save, xs_save, sig_save, ps_save

def get_ts_save(tfin, dt, dt_save):
    if not dt_save:
        saverate = int((1e8 * tfin) / (1e8 * dt))
        ts_save = np.array([0., tfin])
        return ts_save, saverate
    
    saverate = int((1e8 * dt_save) / (1e8 * dt))
    if (1e8 * tfin) % (1e8 * dt_save) == 0:
        nsaves = 1 + int((1e8 * tfin) / (1e8 * dt_save))
        ts_save = np.linspace(0., tfin, nsaves)
    else:
        nsaves = 1 + int((tfin - (tfin % dt_save)) / dt_save)
        ts_save = np.linspace(0., tfin - (tfin % dt_save), nsaves)
    return ts_save, saverate
    