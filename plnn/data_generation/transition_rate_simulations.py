"""Perform simulations for a range of transition rates.

Run a number of landscape simulations using a sigmoidal parameter schedule. 
In each case, fix initial and final parameter values, as well as critical time, 
and vary only the rate of transition, r.
"""

import sys, os, argparse
import numpy as np
from plnn.data_generation.simulate import simulate_landscape, get_landscape_func
from plnn.data_generation.phi_animator import PhiSimulationAnimator

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outdir', type=str, required=True,
                        help="path to output directory")
    parser.add_argument('--index', type=int, default=None, 
                        help="Simulation index to run (for parallel purposes).")

    parser.add_argument('--landscape_name', type=str, required=True,
                        choices=['phi1', 'phi2'])
    parser.add_argument('--sigma', type=float, default=0.01,
                        help="Noise parameter (constant)")
    parser.add_argument('--tcrit', type=float, default=5.0,
                        help="Sigmoid critical time (shared across signals)")
    parser.add_argument('--logr_range', type=float, nargs=2, default=[-2., 2.],
                        help="Range of log(r) values to span.")
    parser.add_argument('--num_rs', type=int, default=21)
    
    parser.add_argument('--nsims', type=int, default=10,
                        help="Number of simulations for each rate value.")
    parser.add_argument('--ncells', type=int, default=500)
    parser.add_argument('--x0', type=float, nargs='+', default=(0, -0.5))
    parser.add_argument('--tfin', type=float, default=10.)
    parser.add_argument('--dt', type=float, default=1e-3)
    parser.add_argument('--dt_save', type=float, default=2.)
    parser.add_argument('--burnin', type=float, default=0.5,
                        help="Burnin time as a factor of the total duration.")
    
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
    index = args.index
    sigma = args.sigma
    tcrit = args.tcrit
    logr_range = args.logr_range
    num_rs = args.num_rs
    landscape_name = args.landscape_name
    nsims = args.nsims
    ncells = args.ncells
    x0 = args.x0
    tfin = args.tfin
    dt = args.dt
    dt_save = args.dt_save
    burnin = int(args.burnin * tfin // dt)
    do_animate = args.animate
    ani_duration = args.duration
    seed = args.seed if args.seed else np.random.randint(2**32)
    
    nsignals = 2
    signal_schedule = 'sigmoid'
    noise_schedule = 'constant'
    param_func_name = 'identity'
    noise_args = [sigma]
    
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
    
    log_rs = np.linspace(*logr_range, num_rs, endpoint=True)

    os.makedirs(outdir, exist_ok=True)
    parent_rng = np.random.default_rng(seed=seed)  # RNG shared across indices
    streams = parent_rng.spawn(num_rs)  # RNG for each value of r

    if index is None or index == 0:
        np.save(f"{outdir}/log_rs.npy", log_rs)

    if index is not None:
        log_rs = [log_rs[index]]

    for i, logr in enumerate(log_rs):
        if index is None:
            rng = streams[i]
            subdir = f"{outdir}/r{i}"
        else:
            rng = streams[index]
            subdir = f"{outdir}/r{index}"
        
        os.makedirs(subdir, exist_ok=True)
        np.savetxt(f"{subdir}/logr.txt", [logr], '%f')
        np.savetxt(f"{subdir}/nsims.txt", [nsims], '%d')

        if do_animate:
            from cont.binary_choice import get_binary_choice_curves
            from cont.binary_flip import get_binary_flip_curves
            if landscape_name == 'phi1':
                landscape_tex = "Binary Choice"
                bifcurves, bifcolors = get_binary_choice_curves()
            elif landscape_name == 'phi2':
                landscape_tex = "Binary Flip"
                bifcurves, bifcolors = get_binary_flip_curves()
        
        # Sample parameter values
        p_initials = []
        p_finals = []
        for simidx in range(nsims):
            p_initial, p_final = sampler(rng)
            p_initials.append(p_initial)
            p_finals.append(p_final)

        # Substreams for each simulation
        sim_subseeds = rng.choice(2**32, size=nsims, replace=False)
        substreams = rng.spawn(nsims)
        for simidx in range(nsims):
            simdir = f"{subdir}/sim{simidx}"
            os.makedirs(simdir, exist_ok=True)

            p_initial, p_final = p_initials[simidx], p_final[simidx]
            
            sigparams = np.array([
                [tcrit, p_initial[0], p_final[0], np.exp(logr)],
                [tcrit, p_initial[1], p_final[1], np.exp(logr)],
            ])

            with open(f"{simdir}/args.txt", 'w') as f:
                f.write(str(args))
            
            sim_rng = substreams[simidx]
            sim_seed = sim_rng.integers(2**32)
            
            ts_saved, xs_saved, sigs_saved, ps_saved = simulate_landscape(
                landscape_name=landscape_name,
                ncells=ncells, 
                x0=x0,
                tfin=tfin, 
                dt=dt, 
                dt_save=dt_save, 
                burnin=burnin,
                nsignals=nsignals,
                signal_schedule=signal_schedule, 
                sigparams=sigparams,
                param_func_name=param_func_name,
                noise_schedule=noise_schedule, 
                noise_args=noise_args,
                rng=np.random.default_rng([sim_seed, sim_subseeds[simidx]]),
            )
            
            # Save output
            np.save(f"{simdir}/ts.npy", ts_saved)
            np.save(f"{simdir}/xs.npy", xs_saved)
            np.save(f"{simdir}/sigs.npy", sigs_saved)
            np.save(f"{simdir}/ps.npy", ps_saved)
            np.save(f"{simdir}/sigparams.npy", sigparams)

        
            # Animate simulations
            if do_animate and simidx in args.sims_to_animate:
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
                        burnin=burnin,
                        nsignals=args.nsignals,
                        signal_schedule=args.signal_schedule, 
                        sigparams=sigparams,
                        param_func_name=param_func_name,
                        noise_schedule=args.noise_schedule, 
                        noise_args=args.noise_args,
                        rng=np.random.default_rng([sim_seed, sim_subseeds[simidx]]),
                    )

                    if args.save_animation_data:
                        np.save(f"{simdir}/ani_ts.npy", ts)
                        np.save(f"{simdir}/ani_xs.npy", xs)
                        np.save(f"{simdir}/ani_sigs.npy", sigs)
                        np.save(f"{simdir}/ani_ps.npy", ps)

                info_str = f"Landscape: {landscape_tex}" + \
                    f"\nSignal type: {signal_schedule}" + \
                    f"\n$T = {tfin:.5g}$" + \
                    f"\n$dt = {dt:.5g}$" + \
                    f"\n$\sigma = {noise_args[0]:.5g}$" + \
                    f"\n$\Delta T = {dt_save:.5g}$" + \
                    f"\n$N_{{cells}} = {ncells:.5g}$" + \
                    f"\n$\mathbf{{x}}_0 = {x0}$" + \
                    f"\nburnin: ${args.burnin:.5g}T$"

                sigparams_str = f"Signal parameters:\n  " + \
                    '\n  '.join([', '.join([f"{x:.3g}" for x in s]) 
                                for s in sigparams])

                animator = PhiSimulationAnimator(
                    ts, xs, sigs, ps, 
                    ts_saved, xs_saved, sigs_saved, ps_saved,
                    xlims=[-4, 4], 
                    ylims=[-4, 4],
                    p0lims=[-2, 2],
                    p1lims=[-2, 2],
                    p0idx=0,
                    p1idx=1,
                    phi_func=get_landscape_func(landscape_name),
                    bifcurves=bifcurves,
                    bifcolors=bifcolors,
                    info_str=info_str,
                    sigparams_str=sigparams_str,
                    grads=None,
                    grad_func=None,
                )

                animator.animate(
                    savepath=f"{simdir}/animation", 
                    duration=args.duration
                )


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
