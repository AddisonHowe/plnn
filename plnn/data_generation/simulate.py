import sys, os, argparse
import warnings
import numpy as np
import jax
import jax.numpy as jnp
from plnn.data_generation.simulator import Simulator
from plnn.data_generation.phi_animator import PhiSimulationAnimator
from plnn.data_generation.signals import get_binary_function
from plnn.data_generation.signals import get_sigmoid_function

"""

"""

def simulate_landscape(
        landscape_name,
        ncells,
        x0,
        tfin,
        dt,
        dt_save,
        burnin,
        nsignals,
        signal_schedule,
        sigparams,
        param_func_name='identity',
        param_func_args=None,
        noise_schedule='constant',
        noise_args=[0.1],
        burnin_signal=None,
        metric_name=None,
        metric_args=None,
        seed=None,
        rng=None,
):
    """Simulate dynamics given a parameterized vector field.

    Args:
        landscape_name (str) : Potential function identifier.
        ncells (int) : Number of cells to simulate.
        x0 (ndarray) : Initial cell position.
        tfin (float) : Simulation end time.
        dt (float) : Step size.
        dt_save (float) : Interval at which to save state.
        burnin (int) : Number of initial burnin steps to take.
        nsignals (int) : Number of signals.
        signal_schedule (str) : Signal schedule identifier.
        sigparams (list) : Parameters defining each signal profile.
        param_func_name (str) : Parameter function identifier.
        noise_schedule (str) : Noise schedule identifier.
        noise_args (list) : Arguments for noise function.
        metric_name (str) : Metric identifier.
        metric_args (list) : Arguments for metric function.
        seed (int) : Random number generator seed. Default None.
        rng (Generator) : Random number generator object. Default None.

    Returns:
        ts (ndarray)  : Saved timepoints.
        xs (ndarray)  : Saved state values.
        sigs (ndarray): Saved signal values.
        ps (ndarray)  : Saved parameter values.
    """
    if rng is None:
        rng = np.random.default_rng(seed=seed)

    # Construct the simulator
    f = get_landscape_field_func(landscape_name)
    signal_func = get_signal_func(nsignals, signal_schedule, sigparams)
    param_func = get_param_func(param_func_name, param_func_args)
    noise_func = get_noise_func(noise_schedule, noise_args)
    metric_func = get_metric_func(metric_name, metric_args)
    simulator = Simulator(f, signal_func, param_func, noise_func, metric_func)
    # Run the simulation
    ts, xs, sigs, ps = simulator.simulate(
        ncells, x0, tfin, dt=dt, burnin=burnin, dt_save=dt_save, 
        burnin_signal=burnin_signal, rng=rng
    )
    return ts, xs, sigs, ps

##########################
##  Simulation Getters  ##
##########################

def get_landscape_func(landscape_name):
    if landscape_name == 'phi1':
        return phi1
    elif landscape_name == 'phi2':
        return phi2
    elif landscape_name == 'phiq':
        return phiq
    elif landscape_name == 'phi_stitched':
        return phi_stitched
    else:
        raise NotImplementedError(f"Unknown landscape: {landscape_name}")

def get_landscape_field_func(landscape_name):
    if landscape_name == 'phi1':
        return phi1_field
    elif landscape_name == 'phi2':
        return phi2_field
    elif landscape_name == 'phiq':
        return phiq_field
    elif landscape_name == 'phi_stitched':
        return phi_stitched_field
    else:
        raise NotImplementedError(f"Unknown landscape: {landscape_name}")

def get_signal_func(nsignals, signal_schedule, sigparams):
    if sigparams.ndim == 1:
        sigparams = np.array(sigparams).reshape([nsignals, -1])
    nsigparams = sigparams.shape[1]
    if signal_schedule == 'binary':
        assert nsigparams == 3, f"Got {nsigparams} args instead of 3."
        tcrit = sigparams[:,0]  # change times
        s0 = sigparams[:,1]     # initial values
        s1 = sigparams[:,2]     # final values
        signal_func = get_binary_function(tcrit, s0, s1)
    elif signal_schedule == 'sigmoid':
        assert nsigparams == 4, f"Got {nsigparams} args instead of 4."
        tcrit = sigparams[:,0]  # change times
        s0 = sigparams[:,1]     # initial values
        s1 = sigparams[:,2]     # final values
        r = sigparams[:,3]      # transition rates
        signal_func = get_sigmoid_function(tcrit, s0, s1, r)
    return signal_func

def get_param_func(param_func_name, param_func_args=None):
    if param_func_name.lower() == "identity":
        return lambda t, s, args: s
    elif param_func_name.lower() == "linear":
        w = param_func_args['weights']
        b = param_func_args['bias']
        def pfunc(t, s, args):
            return s @ w + b
        return pfunc
    else:
        msg = f"Unknown parameter function identifer: {param_func_name}"
        raise NotImplementedError(msg)

def get_noise_func(noise_schedule, noise_args):
    if noise_schedule == 'constant':
        sigma = noise_args[0]
        noise_func = lambda t, x: sigma
    else:
        raise NotImplementedError(f"Unknown noise_schedule: {noise_schedule}")
    return noise_func

def get_metric_func(metric_name, metric_args):
    if metric_name is None:
        return None
    elif metric_name.lower() in ["id", "identity"]:
        dim = metric_args['dim']
        id = jnp.eye(dim, dtype=float)[None,:,:]
        return lambda t, x: jnp.broadcast_to(id, (x.shape[0], dim, dim))
    elif metric_name == 'saddle_v1':
        k1 = metric_args['k1']
        k2 = metric_args['k2']
        def mfunc(t, x):
            x, y = x.T
            g11 = 1 + (k2 * y)**2
            g12 = k1 * k2 * x * y
            g22 = 1 + (k1 * x)**2
            g = jnp.array([
                [g11, g12],
                [g12, g22],
            ]).T.swapaxes(1, 2)
            return g
        return mfunc
    else:
        msg = f"Unknown metric function identifer: {metric_name}"
        raise NotImplementedError(msg)

##########################
##  Gradient Functions  ##
##########################

def phi1(t, xy, p):
    """"""
    x = xy[...,0]
    y = xy[...,1]
    x2 = x*x
    y2 = y*y
    return x2*x2 + y2*y2 + y2*y - 4*x2*y + y2 + p[0]*x + p[1]*y

def phi2(t, xy, p):
    """"""
    x = xy[...,0]
    y = xy[...,1]
    x2 = x*x
    y2 = y*y
    return x2*x2 + y2*y2 + x2*x - 2*x*y2 - x2 + p[0]*x + p[1]*y

def phiq(t, xy, p):
    """Qudaratic Potential with A=1/4, B=1/9."""
    a = 1/4
    b = 1/9
    x = xy[...,0]
    y = xy[...,1]
    return a*x*x + b*y*y + p[0]*x + p[1]*y

def phi_stitched(t, xy, p):
    """"""
    x = xy[...,0]
    vx = 2.
    vy = 1.
    f1 = phi1(t, xy, p[0:2])
    f2 = phi2(t, xy - jnp.array([vx, vy]), p[2:4])
    chi =  (jnp.tanh(10. * x - 5.) + 1) / 2
    return (1-chi) * p[4] * f1 + chi * p[5] * f2

def phi1_field(t, x, p):
    """Vector field of first potential function in Saez et al."""
    return -np.array([
        4*x[:,0]**3 - 8*x[:,0]*x[:,1] + p[0],
        4*x[:,1]**3 + 3*x[:,1]*x[:,1] - 4*x[:,0]*x[:,0] + 2*x[:,1] + p[1]
    ]).T

def phi2_field(t, x, p):
    """Vector field of second potential function in Saez et al."""
    return -np.array([
        4*x[:,0]**3 + 3*x[:,0]*x[:,0] - 2*x[:,1]*x[:,1] - 2*x[:,0] + p[0],
        4*x[:,1]**3 - 4*x[:,0]*x[:,1] + p[1]
    ]).T

def phiq_field(t, x, p):
    """Vector field of Qudaratic Potential with A=1/4, B=1/9."""
    a = 1/4
    b = 1/9
    return -np.array([
        2.*a*x[:,0] + p[0],
        2.*b*x[:,1] + p[1]
    ]).T

def phi_stitched_field(t, xy, p):
    """Vector field of second potential function in Saez et al."""
    x = xy[...,0]
    y = xy[...,1]
    dx = 2.
    dy = 1.
    # vx = x-dx
    # vy = y-dy
    # p1, p2, p3, p4, p5, p6 = p
    # return np.array([
    #     -5. * p6 * (p3*vx - vx**2 + vx**3 + vx**4 + p4*vy - 2*vx*vy**2 + vy**4) / np.cosh(10*x - 5)**2 + 5.*p5 * (-p1*x + x**4 - 4*x**2*y + y*(p2 + y + y**2 + y**3)) / np.cosh(10.*x-5)**2 + 0.5*p6*(-18. + p3 + 34.*x - 21.*x**2 + 4.*x**3 + 4.*y - 2.*y**2) * (np.tanh(5 - 10*x) - 1.) + 0.5*p5*(p1 - 4.*x**3 + 8.*x*y) * (1. + np.tanh(5 - 10*x)),
    #     -p6 * (p4 + 4*vy * (3 - x - 2*y + y**2)) * (0.5 - 0.5 * np.tanh(5 - 10*x)) - p5 * (p2 - 4*x**2 + y*(2 + 3*y + 4*y**2)) * (0.5 + 0.5*np.tanh(5 - 10*x))
    # ]).T
    f1 = phi1_field(t, xy, p[0:2])
    f2 = phi2_field(t, xy - np.array([dx, dy]), p[2:4])
    chi = (np.tanh(10*x - 5) + 1) / 2
    return (1-chi[:,None]) * p[4] * f1 + chi[:,None] * p[5] * f2

#####################
##  Main Function  ##
#####################

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outdir', type=str, required=True)
    parser.add_argument('-t', '--tfin', type=float, default=10)
    parser.add_argument('--dt', type=float, default=1e-2)
    parser.add_argument('--dt_save', type=float, default=1e-1)
    parser.add_argument('-n', '--ncells', type=int, default=100)
    parser.add_argument('--burnin', type=int, default=50)
    parser.add_argument('--landscape_name', type=str, default='phi1')
    parser.add_argument('--nsignals', type=int, default=2)
    parser.add_argument('--signal_schedule', type=str, default='binary',
                        choices=['binary', 'sigmoid'])
    parser.add_argument('--sigparams', type=float, nargs='+', 
                        default=[5, 0, 1, -1, 0])
    parser.add_argument('--noise_schedule', type=str, default='constant')
    parser.add_argument('--noise_args', type=float, nargs='+',
                        default=[0.01])
    parser.add_argument('--x0', type=float, nargs='+', default=(0, -0.5))
    parser.add_argument('--seed', type=int, default=None)
    
    parser.add_argument('--animate', action='store_true')
    parser.add_argument('--duration', type=int, default=10, 
                        help="Duration of animation in seconds")
    parser.add_argument('--animation_dt', type=float, default=None, 
                        help="Timestep between frames in animation")
    return parser.parse_args(args)


def main(args):
    outdir = args.outdir
    duration = args.duration
    seed = args.seed if args.seed else np.random.randint(2**32)

    os.makedirs(outdir, exist_ok=True)
    with open(f"{outdir}/params.txt", 'w') as f:
        f.write(str(args))

    # Run simulation
    ts_saved, xs_saved, sigs_saved, ps_saved = simulate_landscape(
        landscape_name=args.landscape_name,
        ncells=args.ncells, 
        x0=args.x0,
        tfin=args.tfin, 
        dt=args.dt, 
        dt_save=args.dt_save, 
        burnin=args.burnin,
        nsignals=args.nsignals,
        signal_schedule=args.signal_schedule, 
        sigparams=args.sigparams,
        noise_schedule=args.noise_schedule, 
        noise_args=args.noise_args,
        seed=seed,
    )
    
    # Save output
    np.save(f"{outdir}/ts.npy", ts_saved)
    np.save(f"{outdir}/xs.npy", xs_saved)
    np.save(f"{outdir}/sigs.npy", sigs_saved)
    np.save(f"{outdir}/ps.npy", ps_saved)

    # Animate simulation
    if args.animate:
        if args.animation_dt is None:
            ts = ts_saved
            xs = xs_saved
            sigs = sigs_saved
            ps = ps_saved
        else:
            # Process given animation_dt here...
            ani_dt = args.animation_dt

            # Rerun simulation with finer saverate
            ts, xs, sigs, ps = simulate_landscape(
                landscape_name=args.landscape_name,
                ncells=args.ncells, 
                x0=args.x0,
                tfin=args.tfin, 
                dt=args.dt, 
                dt_save=ani_dt, 
                burnin=args.burnin,
                nsignals=args.nsignals,
                signal_schedule=args.signal_schedule, 
                sigparams=args.sigparams,
                noise_schedule=args.noise_schedule, 
                noise_args=args.noise_args,
                seed=seed,
            )

        animator = PhiSimulationAnimator(
            ts, xs, sigs, ps, 
            ts_saved, xs_saved, sigs_saved, ps_saved,
        )
        animator.animate(
            savepath=f"{outdir}/animation", 
            duration=duration
        )


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
