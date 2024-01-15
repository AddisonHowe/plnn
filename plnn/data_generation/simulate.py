import sys, os, argparse
import numpy as np
from plnn.data_generation.simulator import Simulator
from plnn.data_generation.animator import SimulationAnimator

"""

"""

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outdir', type=str, required=True)
    parser.add_argument('-t', '--tfin', type=float, default=10)
    parser.add_argument('--dt', type=float, default=1e-2)
    parser.add_argument('--dt_save', type=float, default=1e-1)
    parser.add_argument('-n', '--ncells', type=int, default=100)
    parser.add_argument('--burnin', type=int, default=50)
    parser.add_argument('--landscape_name', type=str, default='phi1')
    parser.add_argument('--param_schedule', type=str, default='binary')
    parser.add_argument('--param_args', type=float, nargs='+', 
                        default=[5, 0, 1, -1, 0])
    parser.add_argument('--noise_schedule', type=str, default='constant')
    parser.add_argument('--noise_args', type=float, nargs='+',
                        default=[0.01])
    parser.add_argument('--x0', type=float, nargs='+', default=(0, -0.5))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--animate', action='store_true')
    parser.add_argument('--duration', type=int, default=10, 
                        help="Duration of animation in seconds")
    return parser.parse_args(args)


def simulate_landscape(**kwargs):
    """Simulate dynamics of first potential function in Saez et al.
    Args:
        landscape_name () :
        ncells () :
        x0 () :
        tfin () :
        dt () :
        dt_save () :
        burnin () :
        param_schedule () :
        param_args () :
        noise_schedule () :
        noise_args () :
        seed () :
        rng () :
    """
    #~~~~~~~~~~~~  process kwargs  ~~~~~~~~~~~~#
    landscape_name = kwargs.get('landscape_name')
    ncells = kwargs.get('ncells')
    x0 = kwargs.get('x0')
    tfin = kwargs.get('tfin')
    dt = kwargs.get('dt')
    dt_save = kwargs.get('dt_save')
    burnin = kwargs.get('burnin')
    param_schedule = kwargs.get('param_schedule')
    param_args = kwargs.get('param_args')
    noise_schedule = kwargs.get('noise_schedule')
    noise_args = kwargs.get('noise_args')
    seed = kwargs.get('seed', 0)
    rng = kwargs.get('rng', None)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    if rng is None:
        rng = np.random.default_rng(seed=seed)

    # Construct the simulator
    f = get_landscape_func(landscape_name)
    param_func = get_param_func(param_schedule, param_args)
    noise_func = get_noise_func(noise_schedule, noise_args)
    simulator = Simulator(f, param_func, noise_func)
    # Run the simulation
    ts, xs, ps = simulator.simulate(
        ncells, x0, tfin, dt=dt, t0=0,
        burnin=burnin, dt_save=dt_save, rng=rng
    )
    return ts, xs, ps

##########################
##  Simulation Getters  ##
##########################

def get_landscape_func(landscape_name):
    if landscape_name == 'phi1':
        return phi1
    elif landscape_name == 'phi2':
        return phi2
    else:
        raise NotImplementedError(f"Unknown landscape: {landscape_name}")
    
def get_binary_function(tcrit, p0, p1):
    """Return a binary function that changes from p0 to p1 at time tcrit."""
    return lambda t: p0 if t < tcrit else p1

def get_param_func(param_schedule, param_args):
    if param_schedule == 'binary':
        tcrit = param_args[0]
        p0 = np.array([param_args[1], param_args[2]])  # initial parameters
        p1 = np.array([param_args[3], param_args[4]])  # final parameters
        param_func = get_binary_function(tcrit, p0, p1)
    else:
        raise NotImplementedError(f"Unknown param_schedule: {param_schedule}")
    return param_func


def get_noise_func(noise_schedule, noise_args):
    if noise_schedule == 'constant':
        sigma = noise_args[0]
        noise_func = lambda t, x: sigma
    else:
        raise NotImplementedError(f"Unknown noise_schedule: {noise_schedule}")
    return noise_func

##########################
##  Gradient Functions  ##
##########################

def phi1(t, x, p):
    """Landscape dynamics of first potential function in Saez et al."""
    return -np.array([
        4*x[:,0]**3 - 8*x[:,0]*x[:,1] - p[0],
        4*x[:,1]**3 + 3*x[:,1]*x[:,1] - 4*x[:,0]*x[:,0] + 2*x[:,1] + p[1]
    ]).T

def phi2(t, x, p):
    """Landscape dynamics of second potential function in Saez et al."""
    return -np.array([
        4*x[:,0]**3 + 3*x[:,0]*x[:,0] - 2*x[:,1]*x[:,1] - 2*x[:,0] + p[0],
        4*x[:,1]**3 - 4*x[:,0]*x[:,1] + p[1]
    ]).T

#####################
##  Main Function  ##
#####################

def main(args):
    outdir = args.outdir
    duration = args.duration

    os.makedirs(outdir, exist_ok=True)
    with open(f"{outdir}/params.txt", 'w') as f:
        f.write(str(args))

    # Run simulation
    ts, xs = simulate_landscape(
        landscape_name=args.landscape_name,
        ncells=args.ncells, x0=args.x0,
        tfin=args.tfin, dt=args.dt, dt_save=args.dt_save, burnin=args.burnin,
        param_schedule=args.param_schedule, param_args=args.param_args,
        noise_schedule=args.noise_schedule, noise_args=args.noise_args,
        seed=args.seed,
    )
    
    # Save output
    np.save(f"{outdir}/ts.npy", ts)
    np.save(f"{outdir}/xs.npy", xs)

    # Animate simulation
    if args.animate:
        animator = SimulationAnimator(ts, xs)
        animator.animate(savepath=f"{outdir}/animation", duration=duration)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
