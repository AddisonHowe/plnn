import sys, os, argparse
import csv
import numpy as np
from plnn.data_generation.simulate import simulate_landscape, get_landscape_func
# from plnn.data_generation.animator import SimulationAnimator
from plnn.data_generation.phi_animator import PhiSimulationAnimator

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
    
    parser.add_argument('--nsignals', type=int, default=2)
    parser.add_argument('--signal_schedule', type=str,
                        choices=['binary', 'sigmoid'])
    parser.add_argument('--s10_range', type=float, nargs=2, default=[-1, 1])
    parser.add_argument('--s20_range', type=float, nargs=2, default=[-1, 1])
    parser.add_argument('--s11_range', type=float, nargs=2, default=[-1, 1])
    parser.add_argument('--s21_range', type=float, nargs=2, default=[-1, 1])

    parser.add_argument('--param_func', type=str,
                        choices=['identity'])
    
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
    
    parser.add_argument('--seed', type=int, default=None)
    return parser.parse_args(args)


def get_sigparams(
        tfin, 
        s10_range=[-1, 1], 
        s20_range=[-1, 1], 
        s11_range=[-1, 1], 
        s21_range=[-1, 1], 
        rng=None, 
        seed=None
):
    if rng is None:
        rng = np.random.default_rng(seed=seed)
    tcrit = rng.uniform(0, tfin)
    s10 = rng.uniform(*s10_range)
    s20 = rng.uniform(*s20_range)
    s11 = rng.uniform(*s11_range)
    s21 = rng.uniform(*s21_range)
    return np.array([
        [tcrit, s10, s11],
        [tcrit, s20, s21]
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
    nsignals = args.nsignals
    signal_schedule = args.signal_schedule
    s10_range = args.s10_range
    s20_range = args.s20_range
    s11_range = args.s11_range
    s21_range = args.s21_range
    param_func_name = args.param_func
    noise_schedule = args.noise_schedule
    noise_args = args.noise_args
    seed = args.seed if args.seed else np.random.randint(2**32)

    rng = np.random.default_rng(seed=seed)
    os.makedirs(outdir, exist_ok=True)
    
    np.savetxt(f"{outdir}/nsims.txt", [nsims], '%d')

    if args.animate:
        from cont.binary_choice import get_binary_choice_curves
        from cont.binary_flip import get_binary_flip_curves
        if landscape_name == 'phi1':
            bifcurves, bifcolors = get_binary_choice_curves()
        elif landscape_name == 'phi2':
            bifcurves, bifcolors = get_binary_flip_curves()
                
    for nsim in range(nsims):
        simdir = f"{outdir}/sim{nsim}"
        os.makedirs(simdir, exist_ok=True)
        
        sigparams = get_sigparams(
            tfin, 
            s10_range=s10_range, 
            s20_range=s20_range, 
            s11_range=s11_range, 
            s21_range=s21_range, 
            rng=rng
        )

        with open(f"{simdir}/args.txt", 'w') as f:
            f.write(str(args))
        
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
            rng=rng,
        )
        
        # Save output
        np.save(f"{simdir}/ts.npy", ts_saved)
        np.save(f"{simdir}/xs.npy", xs_saved)
        np.save(f"{simdir}/sigs.npy", sigs_saved)
        np.save(f"{simdir}/ps.npy", ps_saved)
        np.save(f"{simdir}/sigparams.npy", sigparams)

        
    # Animate simulations
    if args.animate:
        for simidx in args.sims_to_animate:
            simdir = f"{outdir}/sim{simidx}"
            ts_saved = np.load(f"{simdir}/ts.npy")
            xs_saved = np.load(f"{simdir}/xs.npy")
            sigs_saved = np.load(f"{simdir}/sigs.npy")
            ps_saved = np.load(f"{simdir}/ps.npy")
            sigparams = np.load(f"{simdir}/sigparams.npy")
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
                    sigparams=sigparams,
                    param_func_name=param_func_name,
                    noise_schedule=args.noise_schedule, 
                    noise_args=args.noise_args,
                    seed=seed,
                )

            animator = PhiSimulationAnimator(
                ts, xs, sigs, ps, 
                ts_saved, xs_saved, sigs_saved, ps_saved,
                xlims=[-4, 4], 
                ylims=[-4, 4],
                p0lims=[-2, 2],
                p1lims=[-2, 2],
                p0idx=0,
                p0idxstr='$p_1$',
                p1idx=1,
                p1idxstr='$p_2$',
                phi_func=get_landscape_func(landscape_name),
                bifcurves=bifcurves,
                bifcolors=bifcolors,
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
