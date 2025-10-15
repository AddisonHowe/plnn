"""Deep NN Parameterized Landscape.

"""

import json
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
from jaxtyping import Array, Float
import equinox as eqx

from .plnn import PLNN


class DeepPhiPLNN(PLNN):

    include_phi_bias: bool
    latent_dim: int

    def __init__(
            self, 
            key, dtype=jnp.float32, *,
            include_phi_bias=True,
            phi_hidden_dims=[16, 32, 32, 16],
            phi_hidden_acts='softplus',
            phi_final_act=None,
            phi_layer_normalize=False,
            latent_dim=None,
            **kwargs
    ):
        key, subkey = jrandom.split(key, 2)
        super().__init__(subkey, dtype=dtype, latent_dim=latent_dim, **kwargs)
        
        self.model_type = "deep_phi_plnn"
        self.include_phi_bias = include_phi_bias
        if latent_dim is None:
            self.latent_dim = self.ndims
        else:
            self.latent_dim = latent_dim

        key, subkey = jrandom.split(key, 2)
        self.phi_module = self._construct_phi_module(
            subkey, 
            phi_hidden_dims, 
            phi_hidden_acts, 
            phi_final_act, 
            phi_layer_normalize,
            bias=include_phi_bias, 
            dtype=dtype
        )

    ######################
    ##  Getter Methods  ##
    ######################

    def get_parameters(self) -> dict:
        """Return dictionary of learnable model parameters.

        The returned dictionary contains the following strings as keys: 
            phi.w, phi.b, tilt.w, tilt.b, metric.w, metric.b, sigma
        Each key maps to a list of jnp.array objects.
        
        Returns:
            dict: Dictionary containing learnable parameters.
        """
        d = super().get_parameters()
        phi_linlayers = self._get_linear_module_layers(self.phi_module)
        d.update({
            'phi.w' : [l.weight for l in phi_linlayers],
            'phi.b' : [l.bias for l in phi_linlayers],
        })
        return d
    
    def get_hyperparameters(self) -> dict:
        """Return dictionary of hyperparameters specifying the model.
        
        Returns:
            dict: dictionary of hyperparameters.
        """
        d = super().get_hyperparameters()
        d['include_phi_bias'] = self.include_phi_bias
        return d
    
    def get_linear_layer_parameters(self) -> list[Array]:
        """Return a list of learnable parameters from linear layers.
        
        Args:
            include_metric (bool, optional) : Whether to include metric params.
                Default False.
        Returns:
            list[Array] : List of linear layer learnable parameter arrays.
        """
        params = super().get_linear_layer_parameters()
        phi_linlayers  = self._get_linear_module_layers(self.phi_module)
        phi_params = []
        for layer in phi_linlayers:
            phi_params.append(layer.weight)
            if layer.bias is not None:
                phi_params.append(layer.bias)
        return phi_params + params
    
    def get_info_string(self, usetex=True) -> str:
        s = super().get_info_string(usetex=usetex)
        if not usetex:
            return s
        s += f"\\newline $\\sigma={self.get_sigma():.3g}$"
        tw = self.get_parameters()['tilt.w'][0]
        tb = self.get_parameters()['tilt.b'][0]
        wstr = "\\\\".join(['&'.join([f"{x:.3g}" for x in row]) for row in tw])
        wstr = f"\\begin{{bmatrix}}{wstr}\\end{{bmatrix}}"
        if tb is None:
            bstr = ""
        else:
            bstr = "\\\\".join([f"{x:.3g}" for x in tb])
            bstr = f"+\\begin{{bmatrix}}{bstr}\\end{{bmatrix}}"
        s += f"\\newline $\\boldsymbol{{\\tau}}=\\Psi(\\boldsymbol{{s}})"
        s += f"={wstr}\\boldsymbol{{s}}{bstr}$"
        return s

    ##############################
    ##  Core Landscape Methods  ##
    ##############################
    
    def eval_phi(
        self, 
        y: Float[Array, "ndims"]
    ) -> Float:
        """Evaluate potential value without tilt.

        Args:
            y (Array) : State. Shape (d,).
        Returns:
            Array of shape (1,).
        """
        return self.eval_confinement(y) + self.phi_module(y).squeeze(-1)

    def eval_grad_phi(
        self, 
        t: Float, 
        y: Float[Array, "ndims"]
    ) -> Float[Array, "ndims"]:
        """Evaluate gradient of potential without tilt.

        Args:
            t (Scalar) : Time.
            y (Array)  : State. Shape (d,).
        Returns:
            Array of shape (d,).
        """
        return self.eval_grad_confinement(y) + \
               jax.jacrev(self.phi_module)(y).squeeze(0)
    
    ##########################
    ##  Model Construction  ##
    ##########################

    @staticmethod
    def make_model(
        key, *, 
        dtype=jnp.float32,
        ndims=2, 
        nparams=2,
        nsigs=2, 
        ncells=100, 
        signal_type='jump', 
        nsigparams=3, 
        sigma_init=1e-2, 
        solver='euler', 
        dt0=1e-2, 
        vbt_tol=1e-6,
        confine=False,
        confinement_factor=1.,
        sample_cells=True, 
        include_phi_bias=True, 
        include_tilt_bias=False,
        phi_hidden_dims=[16,32,32,16], 
        phi_hidden_acts='softplus', 
        phi_final_act=None, 
        phi_layer_normalize=False, 
        tilt_hidden_dims=[],
        tilt_hidden_acts=None,
        tilt_final_act=None,
        tilt_layer_normalize=False,
    ) -> tuple['DeepPhiPLNN', dict]:
        """Construct a model and store all hyperparameters.
        
        Args:
            key
            dtype
            ndims
            nparams
            nsigs
            ncells
            signal_type
            nsigparams
            sigma_init
            solver
            dt0
            vbt_tol
            confine
            confinement_factor
            sample_cells
            include_phi_bias
            include_tilt_bias
            include_metric_bias
            phi_hidden_dims
            phi_hidden_acts
            phi_final_act
            phi_layer_normalize
            tilt_hidden_dims
            tilt_hidden_acts
            tilt_final_act
            tilt_layer_normalize
        
        Returns:
            DeepPhiPLNN: Model instance.
            dict: Dictionary of hyperparameters.
        """
        model = DeepPhiPLNN(
            key=key,
            dtype=dtype,
            ndims=ndims, 
            nparams=nparams, 
            nsigs=nsigs, 
            ncells=ncells,
            signal_type=signal_type, 
            nsigparams=nsigparams,
            sigma_init=sigma_init,
            solver=solver, 
            dt0=dt0, 
            vbt_tol=vbt_tol, 
            confine=confine, 
            confinement_factor=confinement_factor, 
            sample_cells=sample_cells,
            include_phi_bias=include_phi_bias,
            include_tilt_bias=include_tilt_bias,
            phi_hidden_dims=phi_hidden_dims, 
            phi_hidden_acts=phi_hidden_acts, 
            phi_final_act=phi_final_act,
            phi_layer_normalize=phi_layer_normalize,
            tilt_hidden_dims=tilt_hidden_dims, 
            tilt_hidden_acts=tilt_hidden_acts, 
            tilt_final_act=tilt_final_act,
            tilt_layer_normalize=tilt_layer_normalize,
        )
        hyperparams = model.get_hyperparameters()
        # Append to dictionary those hyperparams not stored internally.
        hyperparams.update({
            'phi_hidden_dims' : phi_hidden_dims,
            'phi_hidden_acts' : phi_hidden_acts,
            'phi_final_act' : phi_final_act,
            'phi_layer_normalize' : phi_layer_normalize,
            # 'tilt_hidden_dims' : tilt_hidden_dims,
            # 'tilt_hidden_acts' : tilt_hidden_acts,
            # 'tilt_final_act' : tilt_final_act,
            # 'tilt_layer_normalize' : tilt_layer_normalize,
        })
        model = jtu.tree_map(
            lambda x: x.astype(dtype) if eqx.is_array(x) else x, model
        )
        return model, hyperparams

    #############################
    ##  Initialization Method  ##
    #############################
    
    def initialize(self, 
            key, dtype=jnp.float32, *,
            init_phi_weights_method='xavier_uniform',
            init_phi_weights_args=[],
            init_phi_bias_method='constant',
            init_phi_bias_args=[0.],
            **kwargs
            # init_tilt_weights_method='xavier_uniform',
            # init_tilt_weights_args=[],
            # init_tilt_bias_method='constant',
            # init_tilt_bias_args=[0.],
    ) -> 'DeepPhiPLNN':
        """Return an initialized version of the model.

        Args:
            key
            dtype
            init_phi_weights_method
            init_phi_weights_args
            init_phi_bias_method
            init_phi_bias_args
            init_tilt_weights_method
            init_tilt_weights_args
            init_tilt_bias_method
            init_tilt_bias_args
            init_metric_weights_method
            init_metric_weights_args
            init_metric_bias_method
            init_metric_bias_args

        Returns:
            DeepPhiPLNN : Initialized model instance.
        """        
        key, subkey = jrandom.split(key, 2)
        model = super().initialize(
            subkey, dtype=dtype, **kwargs
            # init_tilt_weights_method=init_tilt_weights_method,
            # init_tilt_weights_args=init_tilt_weights_args,
            # init_tilt_bias_method=init_tilt_bias_method,
            # init_tilt_bias_args=init_tilt_bias_args,
        )
        
        # Initialize phi module
        return self._initialize_linear_module(
            key, dtype, model, 'phi_module', 
            init_weights_method=init_phi_weights_method,
            init_weights_args=init_phi_weights_args,
            init_biases_method=init_phi_bias_method,
            init_biases_args=init_phi_bias_args,
            include_biases=self.include_phi_bias,
        )
    
    ############################
    ##  Model Saving/Loading  ##
    ############################

    @staticmethod
    def load(fname:str, dtype=jnp.float32) -> tuple['DeepPhiPLNN', dict]:
        """Load a model from a binary parameter file.
        
        Args:
            fname (str): Parameter file to load.
            dtype : Datatype. Either jnp.float32 or jnp.float64.

        Returns:
            DeepPhiPLNN: Model instance.
            dict: Model hyperparameters.
        """
        with open(fname, "rb") as f:
            hyperparams = json.loads(f.readline().decode())
            model, _ = DeepPhiPLNN.make_model(
                key=jrandom.PRNGKey(0), dtype=dtype, **hyperparams
            )
            return eqx.tree_deserialise_leaves(f, model), hyperparams

    ######################
    ##  Helper Methods  ##
    ######################
    
    def _construct_phi_module(self, key, hidden_dims, hidden_acts, final_act, 
                              layer_normalize, bias=True, dtype=jnp.float32):
        return self._construct_ffn(
            key, self.latent_dim, 1,
            hidden_dims, hidden_acts, final_act, layer_normalize, bias, dtype
        )
