"""Input/Output Functions

This module contains functions that can be helpful to load model instances from
saved parameter files, as well as load metadata collected during the training
process.
"""

import os
import numpy as np
import jax.numpy as jnp
from typing import Tuple

from plnn.models.plnn import PLNN
from plnn.models import DeepPhiPLNN


def load_model_from_directory(
        modeldir,
        subdir="states",
        idx=-1,
        model_class=DeepPhiPLNN,
        file_suffix='.pth',
        dtype=jnp.float64,
) -> Tuple[PLNN, dict, int, str]:
    """Load a PLNN model from a parameter file saved during model training.

    Args:
        modeldir (str) : Path to directory containing output training data.
        subdir (str, optional) : The directory within `modeldir` containing the 
            saved model parameter files. Defaults to 'states'.
        idx (int, str, optional) : The particular index to load. If -1 or 'best'
            the optimal model is loaded. Defaults to -1.
        model_class (PLNN subclass) : Model class being loaded. Should be a 
            subclass of PLNN.
        file_suffix (str, optional) : File suffix. Defaults to '.pth'.
        dtype (jnp.dtype, optional) : Datatype of model to load. Defaults to
            jnp.float64.

    Returns:
        (PLNN) : Loaded model instance.
        (dict) : Dictionary of hyperparameters.    
        (int)  : Index of loaded model.
        (str)  : Model name.
        (str)  : Model filepath.
    """

    if idx == -1 or idx == 'best':
        loss_hist_valid = np.load(f"{modeldir}/validation_loss_history.npy")
        idx = 1 + np.argmin(loss_hist_valid[~np.isnan(loss_hist_valid)])

    file_list = os.listdir(f"{modeldir}/{subdir}")
    fname_components = file_list[0].removesuffix(file_suffix).split("_")
    modelname = "_".join(fname_components[:-1])
    model_fpath = f"{modeldir}/{subdir}/{modelname}_{idx}" + file_suffix
    model, hyperparams = model_class.load(model_fpath, dtype=dtype)
    assert isinstance(model, model_class)
    return model, hyperparams, idx, modelname, model_fpath


def load_model_training_metadata(
        modeldir, 
) -> Tuple[dict, dict]:
    """Load metadata from the model training process, including loss history.

    Args:
        modeldir (str) : Path to directory containing output training data.

    Returns:
        (dict) : Dictionary containing arguments passed to the main script.
        (dict) : Dictionary containing metadata collected during the training
            process. Keys include:
                loss_hist_train : Training loss history.
                loss_hist_valid : Validation loss history.
                sigma_hist : Value of the noise parameter at each epoch.
                learning_rate_hist : Learning rate used at each epoch.
                tilt_weight_hist : History of the weights of the tilt transform.
    """
    loss_hist_train = np.load(f"{modeldir}/training_loss_history.npy")
    loss_hist_valid = np.load(f"{modeldir}/validation_loss_history.npy")
    lr_hist = np.load(f"{modeldir}/learning_rate_history.npy")
    sigma_hist = np.load(f"{modeldir}/sigma_history.npy")
    tilt_weight_hist = np.load(f"{modeldir}/tilt_weights_history.npy")
    logged_args = _load_args_from_log(f"{modeldir}/log_args.txt")
    return_dict = {
        "loss_hist_train" : loss_hist_train,
        "loss_hist_valid" : loss_hist_valid,
        "learning_rate_hist" : lr_hist,
        "sigma_hist" : sigma_hist,
        "tilt_weight_hist" : tilt_weight_hist,
    }
    return logged_args, return_dict


########################
##  Helper Functions  ##
########################


_ARGS_TO_LOAD = {
    'ndims', 'nsigs', 'ncells', 'hidden_dims', 'hidden_acts', 'final_act', 
    'layer_normalize', 'infer_noise', 'sigma', 'dtype', 
    'training_data', 'validation_data', 'nsims_training', 'nsims_validation', 
    'init_phi_bias_args', 'init_phi_bias_method', 
    'init_phi_weights_args', 'init_phi_weights_method', 
    'init_tilt_bias_args', 'init_tilt_bias_method', 
    'init_tilt_weights_args', 'init_tilt_weights_method',
    'loss', 'learning_rate', 'optimizer'
}

def _load_args_from_log(logfilepath, args_to_load=_ARGS_TO_LOAD) -> dict:
    """Load key/value arguments logged in the logfile.

    Args:
        logfilepath (str) : Path to the log file to read from.
        args_to_load (set, optional) : Iterable containing key values to load.
    
    Returns:
        (dict) : Argument dictionary mapping keys to argument value.
    """
    args = {}
    with open(logfilepath, 'r') as f:
        for line in f.readlines():
            line = line[0:-1]  # remove \n
            line = line.split(" : ")  # try to split into key, val pair
            if len(line) == 2:
                key, val = line
                if key in args_to_load:
                    args[key] = eval(val)
    return args

