import sys, os, argparse
import csv
import numpy as np
from plnn.data_generation.simulate import simulate_landscape, get_landscape_func
from plnn.data_generation.phi_animator import PhiSimulationAnimator

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outdir', type=str, required=True,
                        help="path to output directory")
    
    parser.add_argument('--nsims', type=int, default=1, 
                        help="Number of simulations to run.")
    parser.add_argument('--ncells', type=int, default=100)
    parser.add_argument('--x0', type=float, nargs='+', default=(0, -0.5))
    parser.add_argument('--tfin', type=float, default=10.)
    parser.add_argument('--dt', type=float, default=1e-2)
    parser.add_argument('--dt_save', type=float, default=1e-1)
    parser.add_argument('--burnin', type=float, default=0.5,
                        help="Burnin time as a factor of the total duration.")
    parser.add_argument('--burnin_signal', type=float, nargs='+', 
                        default=None,
                        help="Signal to use during burnin phase.")

    parser.add_argument('--landscape_name', type=str, required=True, 
                        choices=['phi1', 'phi2', 'phiq'])
    
    parser.add_argument('--nsignals', type=int, default=2)
    parser.add_argument('--signal_schedule', type=str,
                        choices=['binary', 'sigmoid'])
    parser.add_argument('--s10_range', type=float, nargs=2, default=[-1, 1])
    parser.add_argument('--s20_range', type=float, nargs=2, default=[-1, 1])
    parser.add_argument('--s11_range', type=float, nargs=2, default=[-1, 1])
    parser.add_argument('--s21_range', type=float, nargs=2, default=[-1, 1])
    parser.add_argument('--logr1_range', type=float, nargs=2, default=[-3, 2])
    parser.add_argument('--logr2_range', type=float, nargs=2, default=[-3, 2])
    parser.add_argument('--tcrit_buffer0', type=float, default=0.1)
    parser.add_argument('--tcrit_buffer1', type=float, default=0.1)

    parser.add_argument('--param_func', type=str,
                        choices=['identity'])
    
    parser.add_argument('--noise_schedule', type=str, default='constant',
                        choices=['constant'])
    parser.add_argument('--noise_args', type=float, nargs='+',
                        default=[0.01])
    
    parser.add_argument('--metric_name', type=str, default=None,
                        choices=[None, "identity", "saddle_v1"])
    parser.add_argument('--metric_args', nargs='*', default=[])
        
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


def get_sigparams_binary(
        tfin, 
        s10_range=[-1, 1], 
        s20_range=[-1, 1], 
        s11_range=[-1, 1], 
        s21_range=[-1, 1], 
        tcrit_buffer=0.1,
        rng=None, 
        seed=None
):
    if rng is None:
        rng = np.random.default_rng(seed=seed)
    
    if isinstance(tcrit_buffer, (float, int)):
        tcrit_buffer0 = tcrit_buffer
        tcrit_buffer1 = tcrit_buffer
    elif len(tcrit_buffer) == 2:
        tcrit_buffer0, tcrit_buffer1 = tcrit_buffer
    else:
        raise RuntimeError(f"Cannot handle tcrit_buffer: {tcrit_buffer}")
    
    tcrit1 = rng.uniform(tcrit_buffer0*tfin, (1-tcrit_buffer1)*tfin)
    tcrit2 = rng.uniform(tcrit_buffer0*tfin, (1-tcrit_buffer1)*tfin)
    s10 = rng.uniform(*s10_range)
    s20 = rng.uniform(*s20_range)
    s11 = rng.uniform(*s11_range)
    s21 = rng.uniform(*s21_range)
    return np.array([
        [tcrit1, s10, s11],
        [tcrit2, s20, s21]
    ])


def get_sigparams_sigmoid(
        tfin, 
        s10_range=[-1, 1], 
        s20_range=[-1, 1], 
        s11_range=[-1, 1], 
        s21_range=[-1, 1], 
        logr1_range=[-3, 2], 
        logr2_range=[-3, 2], 
        tcrit_buffer=0.1,
        rng=None, 
        seed=None
):
    if rng is None:
        rng = np.random.default_rng(seed=seed)

    if isinstance(tcrit_buffer, (float, int)):
        tcrit_buffer0 = tcrit_buffer
        tcrit_buffer1 = tcrit_buffer
    elif len(tcrit_buffer) == 2:
        tcrit_buffer0, tcrit_buffer1 = tcrit_buffer
    else:
        raise RuntimeError(f"Cannot handle tcrit_buffer: {tcrit_buffer}")
    
    tcrit1 = rng.uniform(tcrit_buffer0*tfin, (1-tcrit_buffer1)*tfin)
    tcrit2 = rng.uniform(tcrit_buffer0*tfin, (1-tcrit_buffer1)*tfin)
    s10 = rng.uniform(*s10_range)
    s20 = rng.uniform(*s20_range)
    s11 = rng.uniform(*s11_range)
    s21 = rng.uniform(*s21_range)
    r1 = np.exp(rng.uniform(*logr1_range))
    r2 = np.exp(rng.uniform(*logr2_range))
    return np.array([
        [tcrit1, s10, s11, r1],
        [tcrit2, s20, s21, r2]
    ])


def process_metric_args(metric_name, metric_args):
    if len(metric_args) % 2 == 1:
        raise RuntimeError(f"Got odd number of metric arguments: {metric_args}")
    # Compile dictionary mapping argument name to value
    d = {}
    for i in range(len(metric_args) // 2):
        d[metric_args[2*i]] = metric_args[2*i + 1]
    # Checks and conversions
    if metric_name is None or metric_name.lower() == 'none':
        pass
    elif metric_name.lower() in ['id', 'identity']:
        if 'dim' not in d:
            raise RuntimeError("Missing identity metric arg `dim`")
        d['dim'] = int(d['dim'])
    elif metric_name.lower() == 'saddle_v1':
        if ('k1' not in d) or ('k2' not in d):
            raise RuntimeError("Missing saddle_v1 metric args `k1` or `k2`")
        d['k1'] = float(d['k1'])
        d['k2'] = float(d['k2'])
    else:
        raise NotImplementedError(f"Metric `{metric_name}` not implemented.")
    return d


def main(args):
    outdir = args.outdir
    nsims = args.nsims
    x0 = args.x0
    landscape_name = args.landscape_name
    tfin = args.tfin
    dt = args.dt
    dt_save = args.dt_save
    ncells = args.ncells
    burnin = int(args.burnin * tfin // dt)
    burnin_signal = args.burnin_signal
    nsignals = args.nsignals
    signal_schedule = args.signal_schedule
    s10_range = args.s10_range
    s20_range = args.s20_range
    s11_range = args.s11_range
    s21_range = args.s21_range
    logr1_range = args.logr1_range
    logr2_range = args.logr2_range
    tcrit_buffer0 = args.tcrit_buffer0
    tcrit_buffer1 = args.tcrit_buffer1
    param_func_name = args.param_func
    noise_schedule = args.noise_schedule
    noise_args = args.noise_args
    metric_name = args.metric_name
    metric_args = process_metric_args(metric_name, args.metric_args)
    seed = args.seed if args.seed else np.random.randint(2**32)
    do_animate = args.animate

    parent_rng = np.random.default_rng(seed=seed)
    os.makedirs(outdir, exist_ok=True)
    
    np.savetxt(f"{outdir}/nsims.txt", [nsims], '%d')

    if do_animate:
        from cont.binary_choice import get_binary_choice_curves
        from cont.binary_flip import get_binary_flip_curves
        if landscape_name == 'phi1':
            landscape_tex = "Binary Choice"
            xlims = [-4, 4]
            ylims = [-4, 4]
            p0lims = [-2, 2]
            p1lims = [-2, 2]
            bifcurves, bifcolors = get_binary_choice_curves()
        elif landscape_name == 'phi2':
            landscape_tex = "Binary Flip"
            bifcurves, bifcolors = get_binary_flip_curves()
            xlims = [-4, 4]
            ylims = [-4, 4]
            p0lims = [-2, 2]
            p1lims = [-2, 2]
        elif landscape_name == 'phiq':
            landscape_tex = "Quadratic Potential"
            bifcurves, bifcolors = None, None
            xlims = None
            ylims = None
            p0lims = None
            p1lims = None
        

    sim_subseeds = parent_rng.choice(2**32, size=nsims, replace=False)
    streams = parent_rng.spawn(nsims)
    for simidx in range(nsims):
        simdir = f"{outdir}/sim{simidx}"
        os.makedirs(simdir, exist_ok=True)
        with open(f"{simdir}/args.txt", 'w') as f:
            f.write(str(args))
        
        sim_rng = streams[simidx]
        sim_seed = sim_rng.integers(2**32)

        if signal_schedule == "binary":
            sigparams = get_sigparams_binary(
                tfin, 
                s10_range=s10_range, 
                s20_range=s20_range, 
                s11_range=s11_range, 
                s21_range=s21_range, 
                tcrit_buffer=0.1,
                rng=sim_rng
            )
        elif signal_schedule == "sigmoid":
            sigparams = get_sigparams_sigmoid(
                tfin, 
                s10_range=s10_range, 
                s20_range=s20_range, 
                s11_range=s11_range, 
                s21_range=s21_range, 
                logr1_range=logr1_range, 
                logr2_range=logr2_range, 
                tcrit_buffer=(tcrit_buffer0, tcrit_buffer1),
                rng=sim_rng,
            )
        else:
            raise RuntimeError(f"Unknown signal schedule {signal_schedule}")
        
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
            param_func_name=param_func_name,
            noise_schedule=noise_schedule, 
            noise_args=noise_args,
            metric_name=metric_name,
            metric_args=metric_args,
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
                    landscape_name=args.landscape_name,
                    ncells=args.ncells, 
                    x0=args.x0,
                    tfin=args.tfin, 
                    dt=args.dt, 
                    dt_save=ani_dt, 
                    burnin=burnin,
                    burnin_signal=burnin_signal,
                    nsignals=args.nsignals,
                    signal_schedule=args.signal_schedule, 
                    sigparams=sigparams,
                    param_func_name=param_func_name,
                    noise_schedule=args.noise_schedule, 
                    noise_args=args.noise_args,
                    metric_name=metric_name,
                    metric_args=metric_args,
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
                xlims=xlims, 
                ylims=ylims,
                p0lims=p0lims,
                p1lims=p1lims,
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
