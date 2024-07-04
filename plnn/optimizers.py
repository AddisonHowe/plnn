"""Optimizers for model training.

Provides functions:
    get_optimizer_args
    select_optimizer
"""

import optax


def get_optimizer_args(args, nepochs) -> dict:
    """Interface between command line arguments and optimizer construction.

    Move arguments in the command line namespace to a dictionary to be used
    in `select_optimizer`.
    """
    nepochs_warmup = args.nepochs_warmup
    if nepochs_warmup is None:
        nepochs_warmup = 1
    
    nepochs_decay = args.nepochs_decay
    if nepochs_decay is None or nepochs_decay < 0:
        nepochs_decay = nepochs - nepochs_warmup

    return {
        'optimizer'           : args.optimizer,
        'momentum'            : args.momentum,
        'weight_decay'        : args.weight_decay,
        'clip'                : args.clip,
        'lr_schedule'         : args.lr_schedule,
        'learning_rate'       : args.learning_rate,
        'nepochs_warmup'      : nepochs_warmup,
        'nepochs_decay'       : nepochs_decay,
        'final_learning_rate' : args.final_learning_rate,
        'peak_learning_rate'  : args.peak_learning_rate,
        'warmup_cosine_decay_exponent' : args.warmup_cosine_decay_exponent,
    }


def select_optimizer(
        optimization_method, 
        optim_args,
        batch_size,
        dataset_size,
):
    """Build an optimizer with specified learning rate schedule.

    Args:
        optimization_method (str) : Optimization algorithm.
        optim_args (dict)         : Command line args specific to optimization.
        batch_size (int)          : Training batch size.
        dataset_size (int)        : Total number of training samples.
    
    Returns:
        Optimizer object.
    """

    # Get the learning rate schedule.
    schedule = optim_args.get('lr_schedule')

    if schedule == "exponential_decay":
        lr_sched = _get_exponential_decay_schedule(
            optim_args, batch_size, dataset_size
        )
    elif schedule == "warmup_cosine_decay":
        lr_sched = _get_warmup_cosine_decay_schedule(
            optim_args, batch_size, dataset_size
        )
    elif schedule == "constant":
        lr_sched = _get_constant_schedule(optim_args)
    else:
        msg = f"Learning rate schedule {schedule} not yet implemented."
        raise NotImplementedError(msg)
    
    # Construct the optimizer object.
    if optimization_method == 'sgd':
        optimizer = optax.inject_hyperparams(optax.sgd)(
            learning_rate=lr_sched, 
            momentum=optim_args.get('momentum'),
        )
    elif optimization_method == 'adam':
        optimizer = optax.inject_hyperparams(optax.adam)(
            learning_rate=lr_sched, 
        )
    elif optimization_method == 'adamw':
        optimizer = optax.inject_hyperparams(optax.adamw)(
            learning_rate=lr_sched, 
            b1=0.9, 
            b2=0.999, 
            eps=1e-08, 
            eps_root=0.0,
            weight_decay=optim_args.get('weight_decay'),
        )
    elif optimization_method == 'rms':
        optimizer = optax.inject_hyperparams(optax.rmsprop)(
            learning_rate=lr_sched,
            momentum=optim_args.get('momentum'),
            decay=optim_args.get('weight_decay'),
        )
    else:
        msg = f"{optimization_method} optimization not implemented."
        raise RuntimeError(msg)

    # Handle clipping if given, otherwise use a large value as default.
    if optim_args.get('clip') is not None and optim_args.get('clip') > 0:
        optimizer = optax.chain(
            optax.clip(optim_args.get('clip')), 
            optimizer, 
        )
    else:
        optimizer = optax.chain(
            optax.clip(1000.), 
            optimizer, 
        )
    
    return optimizer


#######################################
##  Learning Rate Scheduler Getters  ##
#######################################

def _get_constant_schedule(args):
    return optax.constant_schedule(args['learning_rate'])


def _get_exponential_decay_schedule(
        args, batch_size, training_size,
) -> optax.Schedule:
    initial_learning_rate = args['learning_rate']
    final_learning_rate = args['final_learning_rate']
    transition_begin = args['nepochs_warmup']
    transition_epochs = args['nepochs_decay']
    batches_per_epoch = _get_batches_per_epoch(training_size, batch_size)
    nsteps_begin = transition_begin * batches_per_epoch
    transition_steps = transition_epochs * batches_per_epoch
    decay_rate = final_learning_rate / initial_learning_rate
    return optax.exponential_decay(
            init_value=initial_learning_rate,
            transition_begin=nsteps_begin,
            transition_steps=transition_steps,
            decay_rate=decay_rate,
        )


def _get_warmup_cosine_decay_schedule(
        args, batch_size, training_size,
) -> optax.Schedule:
    init_value = args['learning_rate']
    peak_value = args['peak_learning_rate']
    end_value = args['final_learning_rate']
    nepochs_warmup = args['nepochs_warmup']
    nepochs_decay = args['nepochs_decay']
    batches_per_epoch = _get_batches_per_epoch(training_size, batch_size)
    warmup_steps = nepochs_warmup * batches_per_epoch
    decay_steps = nepochs_decay * batches_per_epoch
    warmup_cosine_decay_exponent = args['warmup_cosine_decay_exponent']
    return optax.warmup_cosine_decay_schedule(
        init_value=init_value,
        peak_value=peak_value,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=end_value,
        exponent=warmup_cosine_decay_exponent,
    )


##########################
##  Timestep Scheduler  ##
##########################

def get_dt_schedule(dt_schedule_name, args) -> optax.Schedule:
    if dt_schedule_name == 'constant':
        return optax.constant_schedule(args['dt'])
    elif dt_schedule_name == 'stepped':
        bounds = args['dt_schedule_bounds']
        scales = args['dt_schedule_scales']
        bounds_and_scales = {b: s for b, s in zip(bounds, scales)}
        return optax.piecewise_constant_schedule(args['dt'], bounds_and_scales)

########################
##  Helper Functions  ##
########################

def _get_batches_per_epoch(training_size, batch_size):
    return int(training_size // batch_size)
