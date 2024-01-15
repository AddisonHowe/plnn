import numpy as np
import torch

class Simulator:

    def __init__(self, f, param_func, noise_func) -> None:
        """
        Args:
            f (callable): deterministic portion of dynamics. Args: t, x, p.
            param_func (callable): parameter function. Arg: t.
            noise_func (callable): stochastic portion of dynamics. Args: t, x.
        """
        self.f = f
        self.param_func = param_func
        self.noise_func = noise_func

    def simulate(self, ncells, x0, tfin, dt=1e-2, t0=0, **kwargs):
        """
        """
        #~~~~~~~~~~~~  process kwargs  ~~~~~~~~~~~~#
        burnin = kwargs.get('burnin', 0)
        dt_save = kwargs.get('dt_save', None)
        rng = kwargs.get('rng', np.random.default_rng())
        device = kwargs.get('device', 'cpu')
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

        if device != 'cpu':
            torchrng = torch.Generator(device=device)
            torchrng.manual_seed(int(rng.integers(100000, 2**32)))

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
        if device != 'cpu':
            xs_save = torch.tensor(xs_save, device=device, dtype=torch.float32)
        
        # Initial parameters
        p0 = self.param_func(t0)
        ps_shape = p0.shape
        ps_save = np.zeros([nsaves, *ps_shape])
        ps_save[0] = p0
        
        # Initialize noise array
        if device == 'cpu':
            dw = np.empty(xs_save[0].shape)
        else:
            dw = torch.empty(xs_save[0].shape, device=device)

        # Initialize x array
        if device == 'cpu':
            x = xs_save[0].copy()
            p = p0.copy()
        else:
            x = xs_save[0].clone().detach().requires_grad_(True)
            p = torch.tensor(p0, device=device, dtype=torch.float32)

        # Burnin steps
        for i in range(burnin):
            if device == 'cpu':
                dw[:] = rng.standard_normal(x.shape)
            else:
                dw[:] = torch.randn(x.shape, generator=torchrng, out=dw)
            term1 = dt * self.f(t0, x, p)
            term2 = dw * self.noise_func(t0, x)
            if device == 'cpu':
                x += term1 + term2
            else:
                with torch.no_grad():
                    x += term1 + term2
                
        # Euler steps
        xs_save[0] = x
        save_counter = 1
        for i, t in zip(range(1, len(ts)), ts[0:-1]):
            if device == 'cpu':
                dw[:] = rng.standard_normal(x.shape)
            else:
                dw[:] = torch.randn(x.shape, generator=torchrng, out=dw)
            p_cpu = self.param_func(t)
            if device == 'cpu':
                p = p_cpu
            else:
                p = torch.tensor(p_cpu, device=device, dtype=torch.float32)
            term1 = dt * self.f(t, x, p)
            term2 = dw * self.noise_func(t, x)
            if device == 'cpu':
                x += term1 + term2
            else:
                with torch.no_grad():
                    x += term1 + term2
            if i % saverate == 0:
                xs_save[save_counter] = x
                ps_save[save_counter] = p_cpu
                save_counter += 1
        
        if device != 'cpu':
            return ts_save, xs_save.cpu().detach().numpy(), ps_save
        else:
            return ts_save, xs_save, ps_save
