import sys, os, argparse
import numpy as np
import jax.numpy as jnp
from plnn.data_generation.simulate import simulate_landscape, get_landscape_func
from plnn.data_generation.phi_animator import PhiSimulationAnimator


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outdir', type=str, required=True,
                        help="path to output directory")
    
    parser.add_argument('--nsims', type=int, default=1, 
                        help="Number of simulations to run.")
    parser.add_argument('--ncells', type=int, default=500)
    parser.add_argument('--x0', type=float, nargs='+', default=(0, -0.5))
    parser.add_argument('--tfin', type=float, default=72.)
    parser.add_argument('--dt', type=float, default=1e-3)
    parser.add_argument('--dt_save', type=float, default=12.)
    parser.add_argument('--burnin', type=float, default=0.05,
                        help="Burnin time as a factor of the total duration.")
    
    parser.add_argument('--landscape_name', type=str, required=True, 
                        choices=['phi_stitched'])
    
    parser.add_argument('--noise_schedule', type=str, default='constant',
                        choices=['constant'])
    parser.add_argument('--noise_args', type=float, nargs='+',
                        default=[0.01])
        
    parser.add_argument('--animate', action='store_true')
    parser.add_argument('--duration', type=int, default=10, 
                        help="Duration of animation in seconds")
    parser.add_argument('--animation_dt', type=float, default=None, 
                        help="Timestep between frames in animation")
    parser.add_argument('--sims_to_animate', type=int, nargs='+', default=[0],
                        help="Indices of the simulations to animate.")
    parser.add_argument('--save_animation_data', action='store_true')
    
    parser.add_argument('--seed', type=int, default=None)
    return parser.parse_args(args)


def get_sigparams_sigmoid(
        tfin, 
        s10_range=[-1, 1], 
        s20_range=[-1, 1], 
        s11_range=[-1, 1], 
        s21_ranges=([-1, 1], [-1, 1]), 
        logr1_range=[-3, 2], 
        logr2_range=[-3, 2], 
        tcrit_buffer=[0., -0.10],
        rng=None, 
        seed=None
):
    if rng is None:
        rng = np.random.default_rng(seed=seed)
    tcrit1 = rng.uniform(tcrit_buffer[0]*tfin, (1-tcrit_buffer[1])*tfin)
    tcrit2 = rng.uniform(tcrit_buffer[0]*tfin, (1-tcrit_buffer[1])*tfin)
    s10 = rng.uniform(*s10_range)
    s20 = rng.uniform(*s20_range)
    s11 = rng.uniform(*s11_range)
    s21 = rng.uniform(*s21_ranges[rng.integers(len(s21_ranges))])
    r1 = np.exp(rng.uniform(*logr1_range))
    r2 = np.exp(rng.uniform(*logr2_range))
    return np.array([
        [tcrit1, s10, s11, r1],
        [tcrit2, s20, s21, r2]
    ])

def main(args):
    outdir = args.outdir
    nsims = args.nsims
    x0 = args.x0
    tfin = args.tfin
    dt = args.dt
    dt_save = args.dt_save
    ncells = args.ncells
    burnin = int(args.burnin * tfin // dt)

    landscape_name = "phi_stitched"
    
    nsignals = 2
    signal_schedule = "sigmoid"
    s10_range = [0.99, 1.0]                     # Initial CHIR
    s20_range = [0.99, 1.0]                     # Initial FGF
    s11_range = [0.00, 0.01]                    # Final CHIR: near 0
    s21_ranges = ([0.0, 0.02], [0.89, 0.91])    # Final FGF: 0 or 0.9
    logr1_range = [5, 6]
    logr2_range = [5, 6]
    noise_schedule = args.noise_schedule
    noise_args = args.noise_args
    seed = args.seed if args.seed else np.random.randint(2**32)
    do_animate = args.animate

    burnin_signal = [0.0, 1.0]


    param_func_args = {}
    param_func_args['weights'] = np.array([
        [0.93673468, -0.552672891, 0.189965361, 
         -0.660020293, -0.488002102, -0.172011631],
        [1.965818467, 8.606555247, 0.114517759, 
         -0.533331098, -0.115242317, -0.643382254]
    ], dtype=np.float64)
    param_func_args['bias'] = np.array([
        -2.208609006, -7.355403139, -1.591722466, 
        0.658224396, 0.738781211, 0.897892629
    ], dtype=np.float64)

    parent_rng = np.random.default_rng(seed=seed)
    os.makedirs(outdir, exist_ok=True)
    
    np.savetxt(f"{outdir}/nsims.txt", [nsims], '%d')

    sim_subseeds = parent_rng.choice(2**32, size=nsims, replace=False)
    streams = parent_rng.spawn(nsims)
    for simidx in range(nsims):
        simdir = f"{outdir}/sim{simidx}"
        os.makedirs(simdir, exist_ok=True)
        with open(f"{simdir}/args.txt", 'w') as f:
            f.write(str(args))
        
        sim_rng = streams[simidx]
        sim_seed = sim_rng.integers(2**32)

        sigparams = get_sigparams_sigmoid(
            tfin, 
            s10_range=s10_range, 
            s20_range=s20_range, 
            s11_range=s11_range, 
            s21_ranges=s21_ranges, 
            logr1_range=logr1_range, 
            logr2_range=logr2_range, 
            tcrit_buffer=[0., -0.10],
            rng=sim_rng
        )
        
        ts_saved, xs_saved, sigs_saved, ps_saved = simulate_landscape(
            landscape_name=landscape_name,
            ncells=ncells, 
            x0=x0,
            tfin=tfin, 
            dt=dt, 
            dt_save=dt_save, 
            burnin=burnin,
            burnin_signal=burnin_signal,
            nsignals=nsignals,
            signal_schedule=signal_schedule, 
            sigparams=sigparams,
            param_func_name='linear',
            param_func_args=param_func_args,
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
                    landscape_name=landscape_name,
                    ncells=ncells, 
                    x0=x0,
                    tfin=tfin, 
                    dt=dt, 
                    dt_save=ani_dt, 
                    burnin=burnin,
                    burnin_signal=burnin_signal,
                    nsignals=nsignals,
                    signal_schedule=signal_schedule, 
                    sigparams=sigparams,
                    param_func_name='linear',
                    param_func_args=param_func_args,
                    noise_schedule=noise_schedule, 
                    noise_args=noise_args,
                    rng=np.random.default_rng([sim_seed, sim_subseeds[simidx]]),
                )

                if args.save_animation_data:
                    np.save(f"{simdir}/ani_ts.npy", ts)
                    np.save(f"{simdir}/ani_xs.npy", xs)
                    np.save(f"{simdir}/ani_sigs.npy", sigs)
                    np.save(f"{simdir}/ani_ps.npy", ps)

            info_str = f"Landscape: {landscape_name}" + \
                f"\nSignal type: {signal_schedule}" + \
                f"\n$T = {tfin:.5g}$" + \
                f"\n$dt = {dt:.5g}$" + \
                f"\n$\\sigma = {noise_args[0]:.5g}$" + \
                f"\n$\\Delta T = {dt_save:.5g}$" + \
                f"\n$N_{{cells}} = {ncells:.5g}$" + \
                f"\n$\\mathbf{{x}}_0 = {x0}$" + \
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
                bifcurves=None,
                bifcolors=None,
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
