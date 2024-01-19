import numpy as np
# import torch

class Simulator:

    def __init__(self, f, signal_func, param_func, noise_func) -> None:
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

    def simulate(
            self, 
            ncells, 
            x0, 
            tfin, 
            dt=1e-2, 
            t0=0, 
            burnin=0, 
            dt_save=None, 
            rng=np.random.default_rng()
    ):
        """
        """
        # Simulation save points: Save every `saverate` steps
        if dt_save:
            ts_save = np.linspace(t0, tfin, 1 + int((tfin - t0) / dt_save))
            saverate = int(dt_save / dt)
        else:
            ts_save = np.array([t0, tfin])
            saverate = int((tfin - t0) / dt)
        nsaves = len(ts_save)

        # Simulation timesteps
        ts = np.linspace(t0, tfin, 1 + int((tfin - t0) / dt))
        
        # Initial state
        x0 = np.array(x0)
        dim = x0.shape[-1]
        xs_save = np.zeros([nsaves, ncells, dim])
        xs_save[0] = x0
        
        # Initial signals
        sig0 = self.signal_func(t0)
        sig_shape = sig0.shape
        sig_save = np.zeros([nsaves, *sig_shape])
        sig_save[0] = sig0

        # Initial parameters
        p0 = self.param_func(sig0)
        ps_shape = p0.shape
        ps_save = np.zeros([nsaves, *ps_shape])
        ps_save[0] = p0
        
        # Initialize noise array
        dw = np.empty(xs_save[0].shape)

        # Initialize state, signal, and parameter arrays
        x = xs_save[0].copy()
        sig = sig0.copy()
        p = p0.copy()

        # Burnin steps: Update only x. Signal and parameters are fixed.
        for i in range(burnin):
            dw[:] = rng.standard_normal(x.shape)
            term1 = dt * self.f(t0, x, p)
            term2 = dw * self.noise_func(t0, x)
            x += term1 + term2
                
        # Euler steps
        xs_save[0] = x
        save_counter = 1
        for i, t in zip(range(1, len(ts)), ts[0:-1]):
            dw[:] = rng.standard_normal(x.shape)
            sig = self.signal_func(t)
            p = self.param_func(sig)
            term1 = dt * self.f(t, x, p)
            term2 = dw * self.noise_func(t, x)
            x += term1 + term2
            if i % saverate == 0:
                xs_save[save_counter] = x
                sig_save[save_counter] = sig
                ps_save[save_counter] = p
                save_counter += 1
        
        return ts_save, xs_save, sig_save, ps_save
