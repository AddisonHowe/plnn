"""General Helper Functions

"""

import os
import numpy as np
import jax.numpy as jnp

from plnn.models.plnn import PLNN
from plnn.models import DeepPhiPLNN


def load_model_from_file(
        modeldir,
        subdir="states",
        idx=-1,
        model_class=DeepPhiPLNN,
        file_suffix='.pth',
        dtype=jnp.float64,
) -> (PLNN, dict, int, str):
    """Load a PLNN model from a parameter file saved during model training.

    Args:
        modeldir (str) : Path to directory containing training output data.
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
    """

    if idx == -1 or idx == 'best':
        loss_hist_valid = np.load(f"{modeldir}/validation_loss_history.npy")
        idx = 1 + np.argmin(loss_hist_valid[~np.isnan(loss_hist_valid)])

    file_list = os.listdir(f"{modeldir}/{subdir}")
    fname_components = file_list[0].removesuffix(file_suffix).split("_")
    modelname = "_".join(fname_components[:-1])
    model_fpath = f"{modeldir}/{subdir}/{modelname}_{idx}" + file_suffix
    model, hyperparams = model_class.load(model_fpath, dtype=dtype)
    return model, hyperparams, idx, modelname
