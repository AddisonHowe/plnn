import sys, os, argparse
import csv
import numpy as np
from plnn.data_generation.simulate import simulate_landscape
from plnn.data_generation.animator import SimulationAnimator

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outdir', type=str, required=True,
                        help="path to output directory")
    
    parser.add_argument('-ns', '--nsims', type=int, default=1)
    parser.add_argument('-nc', '--ncells', type=int, default=100)
    parser.add_argument('--x0', type=float, nargs='+', default=(0, -0.5))
    parser.add_argument('-t', '--tfin', type=float, default=10)
    parser.add_argument('--dt', type=float, default=1e-2)
    parser.add_argument('--dt_save', type=float, default=1e-1)
    parser.add_argument('--burnin', type=int, default=50)

    parser.add_argument('--landscape_name', type=str,
                        choices=['phi1', 'phi2'])
    
    parser.add_argument('--param_schedule', type=str,
                        choices=['binary', 'sigmoid'])
    parser.add_argument('--p10_range', type=float, nargs=2, default=[-1, 1])
    parser.add_argument('--p20_range', type=float, nargs=2, default=[-1, 1])
    parser.add_argument('--p11_range', type=float, nargs=2, default=[-1, 1])
    parser.add_argument('--p21_range', type=float, nargs=2, default=[-1, 1])
    
    parser.add_argument('--noise_schedule', type=str, default='constant',
                        choices=['constant'])
    parser.add_argument('--noise_args', type=float, nargs='+',
                        default=[0.01])
        
    parser.add_argument('--animate', action='store_true')
    parser.add_argument('--duration', type=int, default=10, 
                        help="Duration of animation in seconds")
    
    parser.add_argument('--seed', type=int, default=None)
    return parser.parse_args(args)

def get_param_args(tfin, 
                   p10_range=[-1, 1], p20_range=[-1, 1], 
                   p11_range=[-1, 1], p21_range=[-1, 1], 
                   rng=None, seed=None):
    if rng is None:
        rng = np.random.default_rng(seed=seed)
    tcrit = rng.uniform(0, tfin)
    p10 = rng.uniform(*p10_range)
    p20 = rng.uniform(*p20_range)
    p11 = rng.uniform(*p11_range)
    p21 = rng.uniform(*p21_range)
    return np.array([
        [tcrit, p10, p11],
        [tcrit, p20, p21]
    ])

def main(args):
    outdir = args.outdir
    nsims = args.nsims
    x0 = args.x0
    landscape_name = args.landscape_name
    tfin = args.tfin
    dt = args.dt
    dt_save = args.dt_save
    ncells = args.ncells
    burnin = args.burnin
    param_schedule = args.param_schedule
    noise_schedule = args.noise_schedule
    noise_args = args.noise_args
    p10_range = args.p10_range
    p20_range = args.p20_range
    p11_range = args.p11_range
    p21_range = args.p21_range
    seed = args.seed

    rng = np.random.default_rng(seed=seed)
    os.makedirs(outdir, exist_ok=True)
                
    for nsim in range(nsims):
        simdir = f"{outdir}/sim{nsim}"
        os.makedirs(simdir, exist_ok=True)
        
        param_args = get_param_args(
            tfin, 
            p10_range=p10_range, p20_range=p20_range, 
            p11_range=p11_range, p21_range=p21_range, 
            rng=rng
        )

        with open(f"{simdir}/params.txt", 'w') as f:
            f.write(str(args) + f'\nparam_args: {str(param_args)}')
        
        ts, xs, ps = simulate_landscape(
            landscape_name=landscape_name,
            ncells=ncells, x0=x0,
            tfin=tfin, dt=dt, dt_save=dt_save, burnin=burnin,
            nparams=2,
            param_schedule=param_schedule, param_args=param_args,
            noise_schedule=noise_schedule, noise_args=noise_args,
            rng=rng,
        )
        
        # Save output
        np.save(f"{simdir}/ts.npy", ts)
        np.save(f"{simdir}/xs.npy", xs)
        np.save(f"{simdir}/ps.npy", ps)
        np.save(f"{simdir}/p_params.npy", param_args)

        
    # Animate simulations
    if args.animate:
        for nsim in range(nsims):
            simdir = f"{outdir}/sim{nsim}"
            ts = np.load(f"{simdir}/ts.npy")
            xs = np.load(f"{simdir}/xs.npy")
            ps = np.load(f"{simdir}/ps.npy")
            animator = SimulationAnimator(ts, xs, ps=ps)
            animator.animate(savepath=f"{simdir}/animation", 
                                duration=args.duration)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
