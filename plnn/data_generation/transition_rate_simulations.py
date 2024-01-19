"""Perform simulations for a range of transition rates.

Run a number of landscape simulations using a sigmoidal parameter schedule. 
In each case, fix initial and final parameter values, as well as critical time, 
and vary only the rate of transition, r.
"""

import sys, os, argparse
import numpy as np
from plnn.data_generation.simulate import simulate_landscape
from plnn.data_generation.animator import SimulationAnimator

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outdir', type=str, required=True)
    parser.add_argument('--landscape_name', type=str, required=True,
                        choices=['phi1', 'phi2'])
    parser.add_argument('--index', type=int, default=None)

    parser.add_argument('--nsims', type=int, default=10,
                        help="Number of simulations for each rate value.")
    parser.add_argument('--ncells', type=int, default=500)
    parser.add_argument('--x0', type=float, nargs='+', default=(0, -0.5))
    parser.add_argument('--tfin', type=float, default=10.)
    parser.add_argument('--dt', type=float, default=1e-3)
    parser.add_argument('--dt_save', type=float, default=2.)
    parser.add_argument('--burnin', type=float, default=0.5,
                        help="Burnin time as a factor of the total duration.")

    parser.add_argument('--sigma', type=float, default=0.01,
                        help="Noise parameter (constant)")
    
    parser.add_argument('--animate', action='store_true')
    parser.add_argument('--duration', type=int, default=10, 
                        help="Duration of animation in seconds")
    parser.add_argument('--animation_dt', type=float, default=None, 
                        help="Timestep between frames in animation")
    parser.add_argument('--sims_to_animate', type=int, nargs='+', default=[0],
                        help="Indices of the simulations to animate.")
    
    parser.add_argument('--seed', type=int, default=None)
    return parser.parse_args(args)


def get_sampler1(p_initial, p_final_1, p_final_2, prob):
    def sampler_func(rng):
        p_final = p_final_1 if rng.random() < prob else p_final_2
        return p_initial, p_final
    return sampler_func


def get_sampler2(p_initial_1, p_initial_2, p_final_1, p_final_2, prob):
    def sampler_func(rng):
        p = rng.random()
        p_initial = p_initial_1 if rng.random() < p else p_initial_2
        p_final   = p_final_1   if rng.random() < p else p_final_2
        return p_initial, p_final
    return sampler_func


def main(args):
    outdir = args.outdir
    landscape_name = args.landscape_name
    index = args.index
    nsims = args.nsims
    ncells = args.ncells
    x0 = args.x0
    tfin = args.tfin
    dt = args.dt
    dt_save = args.dt_save
    burnin = int(args.burnin * tfin // dt)
    sigma = args.sigma
    do_animate = args.animate
    ani_duration = args.duration
    seed = args.seed if args.seed else np.random.randint(2**32)
    
    nparams = 2
    tcrit = 5.
    
    if landscape_name == 'phi1':
        pi = [0., 1.]
        pf1 = [ 0.75, 0.]
        pf2 = [-0.75, 0.]
        prob = 0.5
        sampler = get_sampler1(pi, pf1, pf2, prob)
    elif landscape_name == 'phi2':
        pi1 = [-1.0,  0.5]
        pi2 = [-1.0, -0.5]
        pf1 = [-1.0, -0.5]
        pf2 = [-1.0,  0.5]
        prob = 0.5
        sampler = get_sampler2(pi1, pi2, pf1, pf2, prob)
    
    logr_range = [-2, 2]
    num_rs = 21
    
    log_rs = np.linspace(*logr_range, num_rs, endpoint=True)
    param_schedule = 'sigmoid'
    noise_schedule = 'constant'
    noise_args = [sigma]

    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed=seed)
    subseeds = rng.integers(0, 2**32, num_rs)

    if index is None or index == 0:
        np.save(f"{outdir}/log_rs.npy", log_rs)

    if index is not None:
        log_rs = [log_rs[index]]

    for i, logr in enumerate(log_rs):
        if index is None:
            rng = np.random.default_rng(seed=subseeds[i])
            subdir = f"{outdir}/r{i}"
        else:
            rng = np.random.default_rng(seed=subseeds[index])
            subdir = f"{outdir}/r{index}"
        
        os.makedirs(subdir, exist_ok=True)
        np.savetxt(f"{subdir}/logr.txt", [logr], '%f')
        np.savetxt(f"{subdir}/nsims.txt", [nsims], '%d')
        
        for nsim in range(nsims):
            simdir = f"{subdir}/sim{nsim}"
            os.makedirs(simdir, exist_ok=True)

            p_initial, p_final = sampler(rng)
            
            param_args = np.array([
                [tcrit, p_initial[0], p_final[0], np.exp(logr)],
                [tcrit, p_initial[1], p_final[1], np.exp(logr)],
            ])

            with open(f"{simdir}/params.txt", 'w') as f:
                f.write(str(args) + f'\nparam_args: {str(param_args)}')
            
            ts, xs, ps = simulate_landscape(
                landscape_name=landscape_name,
                ncells=ncells, 
                x0=x0,
                tfin=tfin, 
                dt=dt, 
                dt_save=dt_save, 
                burnin=burnin,
                nparams=nparams,
                param_schedule=param_schedule, 
                param_args=param_args,
                noise_schedule=noise_schedule, 
                noise_args=noise_args,
                rng=rng,
            )
            
            # Save output
            np.save(f"{simdir}/ts.npy", ts)
            np.save(f"{simdir}/xs.npy", xs)
            np.save(f"{simdir}/ps.npy", ps)
            np.save(f"{simdir}/p_params.npy", param_args)

        
        # Animate simulations
        if do_animate:
            simdir = f"{subdir}/sim{0}"
            ts = np.load(f"{simdir}/ts.npy")
            xs = np.load(f"{simdir}/xs.npy")
            ps = np.load(f"{simdir}/ps.npy")
            animator = SimulationAnimator(ts, xs, ps=ps)
            animator.animate(savepath=f"{subdir}/animation", 
                             duration=ani_duration)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
