"""Main script.

"""

import os, sys
import argparse
from datetime import datetime
import numpy as np

import torch
import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax

from plnn.dataset import get_dataloaders
from plnn.models import DeepPhiPLNN, GMMPhiPLNN
from plnn.loss_functions import select_loss_function
from plnn.optimizers import get_optimizer_args, select_optimizer
from plnn.model_training import train_model


def parse_args(args):
    parser = argparse.ArgumentParser()
    
    # Training options
    parser.add_argument('--name', type=str, default="model")
    parser.add_argument('-o', '--outdir', type=str, 
                        default="out/model_training")
    parser.add_argument('-t', '--training_data', type=str, required=True)
    parser.add_argument('-v', '--validation_data', type=str, required=True)
    parser.add_argument('--model_type', type=str, default="deep_phi",
                        choices=['deep_phi', 'gmm_phi'])
    parser.add_argument('-nt', '--nsims_training', type=int, default=None)
    parser.add_argument('-nv', '--nsims_validation', type=int, default=None)
    parser.add_argument('-e', '--num_epochs', type=int, default=50)
    parser.add_argument('-b', '--batch_size', type=int, default=25)
    parser.add_argument('--report_every', type=int, default=10)

    # Model simulation
    parser.add_argument('-nd', '--ndims', type=int, default=2,
                        help="Number of state space dimensions for the data.")
    parser.add_argument('-np', '--nparams', type=int, default=2, 
                        help="Number of landscape parameters.")
    parser.add_argument('-ns', '--nsigs', type=int, default=2, 
                        help="Number of signals in the system.")
    parser.add_argument('-nc', '--ncells', type=int, default=100, 
                        help="Number of cells to evolve internally.")
    parser.add_argument('-dt', '--dt', type=float, default=1e-3,
                        help="Euler timestep to use internally.")
    parser.add_argument('--signal_function', type=str, default='jump',
                        choices=['jump', 'sigmoid'], 
                        help="Identifier for the signal function.")
    parser.add_argument('--solver', type=str, default='euler',
                        choices=['euler', 'reversible_heun', 
                                 'ito_milstein', 'heun'], 
                        help="Internal differential equation solver to use.")    

    # Model architecture
    grp_ma = parser.add_argument_group(
        title="model architecture", 
        description="Specifications of the model structure."
    )
    grp_ma.add_argument('--confine', action="store_true")
    grp_ma.add_argument('--phi_hidden_dims', type=int, nargs='+', 
                        default=[16, 32, 32, 16])
    grp_ma.add_argument('--phi_hidden_acts', type=str, nargs='+', 
                        default=4*['softplus'])
    grp_ma.add_argument('--phi_final_act', type=str, default='softplus')
    grp_ma.add_argument('--phi_layer_normalize', action='store_true')

    grp_ma.add_argument('--tilt_hidden_dims', type=int, nargs='+', 
                        default=[])
    grp_ma.add_argument('--tilt_hidden_acts', type=str, nargs='+', 
                        default=None)
    grp_ma.add_argument('--tilt_final_act', type=str, default=None)
    grp_ma.add_argument('--tilt_layer_normalize', action='store_true')

    grp_ma.add_argument('--infer_metric', action="store_true")
    grp_ma.add_argument('--metric_hidden_dims', type=int, nargs='+', 
                        default=[8, 8, 8, 8])
    grp_ma.add_argument('--metric_hidden_acts', type=str, nargs='+', 
                        default=4*['softplus'])
    grp_ma.add_argument('--metric_final_act', type=str, default=None)
    grp_ma.add_argument('--metric_layer_normalize', action='store_true')

    grp_ma.add_argument('--fix_noise', action="store_true",
                        help="NOT IMPLEMENTED! Fix the model noise parameter.")
    grp_ma.add_argument('--sigma', type=float, default=1e-3,
                        help="Noise level if not inferring sigma. Otherwise, \
                            the initial value for the sigma parameter.")    

    # Model initialization
    parser.add_argument('--init_phi_weights_method', type=str, 
                        default='xavier_uniform', 
                        choices=[None, 'xavier_uniform', 'constant', 'normal'])
    parser.add_argument('--init_phi_weights_args', type=float, nargs='*', 
                        default=[])
    parser.add_argument('--init_phi_bias_method', type=str, 
                        default='constant', 
                        choices=[None, 'constant', 'normal'])
    parser.add_argument('--init_phi_bias_args', type=float, nargs='*', 
                        default=0.)
    parser.add_argument('--init_tilt_weights_method', type=str, 
                        default='xavier_uniform', 
                        choices=[None, 'xavier_uniform', 'constant', 'normal'])
    parser.add_argument('--init_tilt_weights_args', type=float, nargs='*', 
                        default=[])
    parser.add_argument('--init_tilt_bias_method', type=str, 
                        default=None, 
                        choices=[None, 'constant', 'normal'])
    parser.add_argument('--init_tilt_bias_args', type=float, nargs='*', 
                        default=None)
    parser.add_argument('--init_metric_weights_method', type=str, 
                        default='xavier_uniform', 
                        choices=[None, 'xavier_uniform', 'constant', 'normal'])
    parser.add_argument('--init_metric_weights_args', type=float, nargs='*', 
                        default=[])
    parser.add_argument('--init_metric_bias_method', type=str, 
                        default=None, 
                        choices=[None, 'constant', 'normal'])
    parser.add_argument('--init_metric_bias_args', type=float, nargs='*', 
                        default=None)

    # Loss function
    parser.add_argument('--loss', type=str, default="kl", 
                        choices=['kl', 'mcd'], 
                        help='kl: KL divergence est; mcd: mean+cov difference.')
    parser.add_argument('--continuation', type=str, default=None, 
                        help="Path to file with model parameters to load.")
    
    # Optimizer
    grp_op = parser.add_argument_group(
        title="optimization args",
        description="Specifications of the optimization training algorithm."
    )
    grp_op.add_argument('--optimizer', type=str, default="sgd", 
                        choices=['sgd', 'adam', 'rms'])
    grp_op.add_argument('--momentum', type=float, default=0.9)
    grp_op.add_argument('--weight_decay', type=float, default=0.)
    grp_op.add_argument('--clip', type=float, default=1.0)
    grp_op.add_argument('--lr_schedule', type=str, default='exponential_decay',
                        choices=['constant', 'exponential_decay', 
                                 'warmup_cosine_decay'])
    grp_op.add_argument('--learning_rate', type=float, default=1e-3)
    grp_op.add_argument('--nepochs_warmup', type=int, default=1,
                        help="Number of warmup epochs. Applicable to \
                            schedules: exponential_decay.")
    grp_op.add_argument('--nepochs_decay', type=int, default=-1,
                        help="Number of epochs over which to transition. \
                            Applicable to schedules: exponential_decay, \
                                warmup_cosine_decay.")
    grp_op.add_argument('--final_learning_rate', type=float, default=0.001,
                        help="Value to which the learning rate will decay. \
                            Applicable to schedules: exponential_decay, \
                                warmup_cosine_decay.")
    grp_op.add_argument('--peak_learning_rate', type=float, default=0.02,
                        help="Peak value of the learning rate. \
                            Applicable to schedules: warmup_cosine_decay.")
    grp_op.add_argument('--warmup_cosine_decay_exponent', type=float, 
                        default=1.0,
                        help="Exponent in the warmup cosine decay schedule")
    
    # Misc. options
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--dtype', type=str, default="float32", 
                        choices=['float32', 'float64'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--timestamp', action="store_true",
                        help="Add timestamp to out directory.")
    parser.add_argument('--save_all', action="store_true")
    parser.add_argument('--enforce_gpu', action="store_true")

    return parser.parse_args(args)


def main(args):
    outdir = args.outdir
    model_name = args.name
    datdir_train = args.training_data
    datdir_valid = args.validation_data
    nsims_train = args.nsims_training if args.nsims_training else read_nsims(datdir_train)
    nsims_valid = args.nsims_validation if args.nsims_validation else read_nsims(datdir_valid)
    model_type = args.model_type
    ndims = args.ndims
    nparams = args.nparams
    nsigs = args.nsigs
    dt = args.dt
    ncells = args.ncells
    confine = args.confine
    phi_hidden_dims = args.phi_hidden_dims
    phi_hidden_acts = args.phi_hidden_acts
    phi_final_act = args.phi_final_act
    phi_layer_normalize = args.phi_layer_normalize
    tilt_hidden_dims = args.tilt_hidden_dims
    tilt_hidden_acts = args.tilt_hidden_acts
    tilt_final_act = args.tilt_final_act
    tilt_layer_normalize = args.tilt_layer_normalize
    infer_metric = args.infer_metric
    metric_hidden_dims = args.metric_hidden_dims
    metric_hidden_acts = args.metric_hidden_acts
    metric_final_act = args.metric_final_act
    metric_layer_normalize = args.metric_layer_normalize
    init_phi_weights_method = args.init_phi_weights_method
    init_phi_weights_args = args.init_phi_weights_args
    init_phi_bias_method = args.init_phi_bias_method
    init_phi_bias_args = args.init_phi_bias_args
    init_tilt_weights_method = args.init_tilt_weights_method
    init_tilt_weights_args = args.init_tilt_weights_args
    init_tilt_bias_method = args.init_tilt_bias_method
    init_tilt_bias_args = args.init_tilt_bias_args
    init_metric_weights_method = args.init_metric_weights_method
    init_metric_weights_args = args.init_metric_weights_args
    init_metric_bias_method = args.init_metric_bias_method
    init_metric_bias_args = args.init_metric_bias_args
    fix_noise = args.fix_noise  # not implemented
    sigma = args.sigma
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    optimization_method = args.optimizer
    cont_path = args.continuation
    loss_fn_key = args.loss
    signal_function_key = args.signal_function
    solver = args.solver
    seed = args.seed
    dtype = jnp.float32 if args.dtype == 'float32' else jnp.float64
    do_plot = args.plot

    if args.timestamp:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        outdir = outdir + "_" + timestamp

    os.makedirs(outdir, exist_ok=True)
    logfpath = f"{outdir}/log.txt"

    def logprint(s, end='\n', flush=True):
        print(s, end=end, flush=flush)
        with open(logfpath, 'a') as f:
            f.write(s + end)
    
    def log_and_raise_runtime_error(msg):
        logprint("ERROR:\n" + msg)
        raise RuntimeError(msg)

    print(f"Writing to logfile: {logfpath}", flush=True)
    
    logprint("Args:")
    logprint(str(args) + "\n")
    
    # Type test
    testarray = jnp.ones([2, 2], dtype=dtype)
    if testarray.dtype != dtype:
        msg = f"Test array is not of type {dtype}. Got {testarray.dtype}."
        log_and_raise_runtime_error(msg)
        
    # GPU test
    if args.enforce_gpu:
        try:
            _ = jax.device_put(jnp.ones(1), device=jax.devices('gpu')[0])
        except RuntimeError as e:
            msg = f"Could not utilize GPU as requested:\n{e}"
            log_and_raise_runtime_error(msg)
    
    # Select model type
    if model_type == 'deep_phi':
        model_class = DeepPhiPLNN
    elif model_type == 'gmm_phi':
        model_class = GMMPhiPLNN
    else:
        msg = f"Unknown model class {model_type}."
        log_and_raise_runtime_error(msg)
        
    # Handle random number generators and seeds
    seed = seed if seed else np.random.randint(2**32)
    rng = np.random.default_rng(seed=seed)
    torch.manual_seed(int(rng.integers(2**32)))
    key = jrandom.PRNGKey(int(rng.integers(2**32)))
    key, modelkey, initkey, trainkey = jrandom.split(key, 4)
    logprint(f"Using seed: {seed}")

    if cont_path: 
        logprint(f"Continuing training of model {cont_path}")

    # Get training and validation dataloaders
    train_dataloader, valid_dataloader, train_dset, _ = get_dataloaders(
        datdir_train, datdir_valid, nsims_train, nsims_valid, 
        batch_size_train=batch_size, batch_size_valid=batch_size, 
        ndims=ndims, dtype=dtype, return_datasets=True,
    )

    # Get signal specification
    signal_type, nsigparams = get_signal_spec(signal_function_key)
    
    # Load previous model or construct new model
    if cont_path:
        model, hyperparams = model_class.load(cont_path, dtype=dtype)
    else:
        # TODO: This may only work properly for a DeepPhiPLNN.
        # Either add a construction method to the PLNN parent class, or 
        # use a separate make function for each model type. Same may apply for
        # model initialization?
        model, hyperparams = model_class.make_model(
            key=modelkey,
            dtype=dtype,
            ndims=ndims, 
            nparams=nparams, 
            nsigs=nsigs, 
            ncells=ncells, 
            sigma_init=sigma,
            signal_type=signal_type,
            nsigparams=nsigparams,
            confine=confine,
            phi_hidden_dims=phi_hidden_dims,
            phi_hidden_acts=phi_hidden_acts,
            phi_final_act=phi_final_act,
            phi_layer_normalize=phi_layer_normalize,
            tilt_hidden_dims=tilt_hidden_dims,
            tilt_hidden_acts=tilt_hidden_acts,
            tilt_final_act=tilt_final_act,
            tilt_layer_normalize=tilt_layer_normalize,
            metric_hidden_dims=metric_hidden_dims,
            metric_hidden_acts=metric_hidden_acts,
            metric_final_act=metric_final_act,
            metric_layer_normalize=metric_layer_normalize,
            include_phi_bias=True,
            include_tilt_bias=False,
            include_metric_bias=True,
            sample_cells=True,
            infer_metric=infer_metric,
            dt0=dt,
            solver=solver,
        )

        model = model.initialize(
            initkey,
            dtype=dtype,
            init_phi_weights_method=init_phi_weights_method,
            init_phi_weights_args=init_phi_weights_args,
            init_phi_bias_method=init_phi_bias_method,
            init_phi_bias_args=init_phi_bias_args,
            init_tilt_weights_method=init_tilt_weights_method,
            init_tilt_weights_args=init_tilt_weights_args,
            init_tilt_bias_method=init_tilt_bias_method,
            init_tilt_bias_args=init_tilt_bias_args,
            init_metric_weights_method=init_metric_weights_method,
            init_metric_weights_args=init_metric_weights_args,
            init_metric_bias_method=init_metric_bias_method,
            init_metric_bias_args=init_metric_bias_args,
        )

    # Get the loss function
    loss_fn = select_loss_function(loss_fn_key)

    # Optimizer construction
    optimizer_args = get_optimizer_args(args, num_epochs)
    optimizer = select_optimizer(
        optimization_method, optimizer_args,
        batch_size=batch_size, dataset_size=len(train_dset),
    )
    
    if do_plot:
        os.makedirs(f"{outdir}/images", exist_ok=True)

    log_args(outdir, args)
    log_model(outdir, model)

    # Train the model
    train_model(
        model,
        loss_fn, 
        optimizer,
        train_dataloader, 
        valid_dataloader,
        key=trainkey,
        num_epochs=num_epochs,
        batch_size=batch_size,
        model_name=model_name,
        hyperparams=hyperparams,
        outdir=outdir,
        save_all=args.save_all,
        plotting=do_plot,
        plotting_opts={},  # Default plotting options
        report_every=args.report_every,
        logprint=logprint,
        error_raiser=log_and_raise_runtime_error,
    )


########################
##  Helper Functions  ##
########################

def get_signal_spec(key):
    if key == 'jump':
        nsigparams = 3
    elif key == 'sigmoid':
        nsigparams = 4
    else:
        msg = f"Unknown signal function identifier {key}."
        raise RuntimeError(msg)
    return key, nsigparams


def log_args(outdir, args):
    with open(f"{outdir}/log_args.txt", 'w') as f:
        for arg, value in sorted(vars(args).items()):
            f.write(str(arg) + " : " + "%r" % value + "\n")


def log_model(outdir, model):
    np.savetxt(f"{outdir}/ncells.txt", [model.get_ncells()])
    np.savetxt(f"{outdir}/sigma.txt", [model.get_sigma()])
    with open(f"{outdir}/log_model.txt", 'w') as f:
        for arg, value in sorted(vars(model).items()):
            f.write(str(arg) + " : " + "%r" % value + "\n")


def read_nsims(datdir):
    return np.genfromtxt(f"{datdir}/nsims.txt", dtype=int)


#####################
##  Main Entrance  ##
#####################

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    print('args:', args, flush=True)
    main(args)
