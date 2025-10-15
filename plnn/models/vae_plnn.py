"""Variational Autoencoder PLNN

"""

import json
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
from jaxtyping import Array, Float
import equinox as eqx

from .plnn_deep import DeepPhiPLNN

##############################################################################
####################  Abstract AutoEncodedPLNN Base Class  ###################
##############################################################################

class VAEPLNN(DeepPhiPLNN):

    encoder_mlp: eqx.Module
    encoder_mu: eqx.Module
    encoder_logvar: eqx.Module
    decoder: eqx.Module
    
    latent_dim: int
    include_enc_bias: bool
    include_dec_bias: bool

    def __init__(
            self, 
            key, dtype=jnp.float32, *,
            latent_dim: int,
            include_enc_bias,
            enc_hidden_dims,
            enc_hidden_acts,
            enc_layer_normalize,
            include_dec_bias,
            dec_hidden_dims,
            dec_hidden_acts,
            dec_final_act,
            dec_layer_normalize,
            **kwargs
    ):
        key, subkey = jrandom.split(key, 2)
        super().__init__(subkey, dtype=dtype, latent_dim=latent_dim, **kwargs)
        
        self.model_type = "vae_plnn"
        self.latent_dim = latent_dim
        self.include_enc_bias = include_enc_bias
        self.include_dec_bias = include_dec_bias

        key, key_enc, key_dec = jax.random.split(key, 3)

        # Encoder Neural Network: Maps ndims to latent_dim
        enc_mlp, enc_mu, enc_logvar = self._construct_encoder_module(
            key_enc, 
            enc_hidden_dims, 
            enc_hidden_acts, 
            enc_layer_normalize,
            bias=include_enc_bias, 
            dtype=dtype
        )
        self.encoder_mlp = enc_mlp
        self.encoder_mu = enc_mu
        self.encoder_logvar = enc_logvar

        # Decoder Neural Network: Maps ndims to latent_dim
        self.decoder = self._construct_decoder_module(
            key_dec, 
            dec_hidden_dims, 
            dec_hidden_acts, 
            dec_final_act, 
            dec_layer_normalize,
            bias=include_dec_bias, 
            dtype=dtype
        )
    
    #######################
    ##  Forward Methods  ##
    #######################

    def __call__(
        self, 
        t0: Float[Array, "b"],
        t1: Float[Array, "b"],
        y0: Float[Array, "b ncells ndims"],
        sigparams: Float[Array, "b nsigs nsigparams"],
        key: Array,
        return_all: bool = False,
    ) -> Float[Array, "b ncells ndims"]:
        """Forward call. Acts on batched data.

        Simulate across batches a number of initial conditions between times t0
        and t1, where t0 and t1 are arrays.
        
        Args:
            t0 (Array) : Initial times. Shape (b,).
            t1 (Array) : End times. Shape (b,).
            y0 (Array) : Initial conditions. Shape (b,n,d).
            sigparams (Array) : Signal parameters. Shape (b,nsigs,nsigparams).
            key (Array) : PRNGKey.

        Returns:
            (Array) Final states, across batches. Shape (b,n,d).
        """
        # Parse the inputs
        fwdvec = jax.vmap(self.simulate_ensemble, 0)
        encvec = jax.vmap(self.encode_ensemble, 0)
        decvec = jax.vmap(self.decode_ensemble, 0)
        key, sample_key = jrandom.split(key, 2)
        if self.sample_cells and self.ncells > 1:
            y0 = self._sample_y0(sample_key, y0)
        key, enc_key = jax.random.split(key, 2)
        batch_enc_keys = jax.random.split(enc_key, t0.shape[0])
        batch_keys = jax.random.split(key, t0.shape[0])
        
        # Encode the inputs
        z0, z0_mu, z0_logvar = encvec(y0, batch_enc_keys)
        # Decode the inputs
        y0hat = decvec(z0)
        # Evolve in latent space
        z1 = fwdvec(t0, t1, z0, sigparams, batch_keys)
        # Decode the evolved inputs
        y1 = decvec(z1)
        all_outputs = {
            "z0": z0,
            "z0": z0,
            "z0_mu": z0_mu,
            "z0_logvar": z0_logvar,
            "y0hat": y0hat,
            "z1": z1,
        }
        if return_all:
            return y1, all_outputs
        return y1
    
    ######################
    ##  Getter Methods  ##
    ######################

    def get_parameters(self) -> dict:
        """Return dictionary of learnable model parameters.

        The returned dictionary contains the following strings as keys: 
            encoder.w, encoder.b, decoder.w, decoder.b
        Each key maps to a list of jnp.array objects.
        
        Returns:
            dict: Dictionary containing learnable parameters.
        """
        d = super().get_parameters()
        enc_mlp_linlayers = self._get_linear_module_layers(self.encoder_mlp)
        dec_linlayers = self._get_linear_module_layers(self.decoder)
        d.update({
            'encoder_mlp.w' : [l.weight for l in enc_mlp_linlayers],
            'encoder_mlp.b' : [l.bias for l in enc_mlp_linlayers],
            'encoder_mu.w' : self.encoder_mu.weight,
            'encoder_mu.b' : self.encoder_mu.bias,
            'encoder_logvar.w' : self.encoder_logvar.weight,
            'encoder_logvar.b' : self.encoder_logvar.bias,
            'decoder.w' : [l.weight for l in dec_linlayers],
            'decoder.b' : [l.bias for l in dec_linlayers],
        })
        return d
    
    def get_hyperparameters(self) -> dict:
        """Return dictionary of hyperparameters specifying the model.
        
        Returns:
            dict: dictionary of hyperparameters.
        """
        d = super().get_hyperparameters()
        d['latent_dim'] = self.latent_dim
        d['include_enc_bias'] = self.include_enc_bias
        d['include_dec_bias'] = self.include_dec_bias
        return d
    
    def get_linear_layer_parameters(self) -> list[Array]:
        """Return a list of learnable parameters from linear layers.
        
        Returns:
            list[Array] : List of linear layer learnable parameter arrays.
        """
        params = super().get_linear_layer_parameters()
        enc_linlayers = self._get_linear_module_layers(self.encoder_mlp)
        enc_linlayers += self._get_linear_module_layers(self.encoder_mu)
        enc_linlayers += self._get_linear_module_layers(self.encoder_logvar)
        enc_params = []
        for layer in enc_linlayers:
            enc_params.append(layer.weight)
            if layer.bias is not None:
                enc_params.append(layer.bias)
        dec_linlayers  = self._get_linear_module_layers(self.decoder)
        dec_params = []
        for layer in dec_linlayers:
            dec_params.append(layer.weight)
            if layer.bias is not None:
                dec_params.append(layer.bias)
        return params + enc_params + dec_params
    
    ###########################
    ##  Autoencoder Methods  ##
    ###########################

    def encode(self, y, key):
        w = self.encoder_mlp(y)
        mean = self.encoder_mu(w)
        logvar = self.encoder_logvar(w)
        std = jnp.exp(0.5 * logvar)
        noise = jax.random.normal(key, mean.shape)
        z = mean + std * noise
        return z, mean, logvar
    
    def decode(self, z):
        return self.decoder(z)
    
    def encode_ensemble(self, ys, key):
        keys = jrandom.split(key, ys.shape[0])
        zs, means, logvars = jax.vmap(self.encode, 0)(ys, keys)
        return zs, means, logvars
    
    def decode_ensemble(self, zs):
        return jax.vmap(self.decoder, 0)(zs)

    ##############################
    ##  Core Landscape Methods  ##
    ##############################

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
        latent_dim=2,
        include_enc_bias=True,
        enc_hidden_dims=[],
        enc_hidden_acts=None,
        enc_layer_normalize=False,
        include_dec_bias=True,
        dec_hidden_dims=[],
        dec_hidden_acts=None,
        dec_final_act=None,
        dec_layer_normalize=False,
    ) -> tuple['VAEPLNN', dict]:
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
            phi_hidden_dims
            phi_hidden_acts
            phi_final_act
            phi_layer_normalize
            tilt_hidden_dims
            tilt_hidden_acts
            tilt_final_act
            tilt_layer_normalize
            latent_dim
            include_enc_bias
            enc_hidden_dims
            enc_hidden_acts
            enc_layer_normalize
            include_dec_bias
            dec_hidden_dims
            dec_hidden_acts
            dec_final_act
            dec_layer_normalize
        
        Returns:
            VAEPLNN: Model instance.
            dict: Dictionary of hyperparameters.
        """
        model = VAEPLNN(
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
            latent_dim=latent_dim,
            include_enc_bias=include_enc_bias,
            enc_hidden_dims=enc_hidden_dims,
            enc_hidden_acts=enc_hidden_acts,
            enc_layer_normalize=enc_layer_normalize,
            include_dec_bias=include_dec_bias,
            dec_hidden_dims=dec_hidden_dims,
            dec_hidden_acts=dec_hidden_acts,
            dec_final_act=dec_final_act,
            dec_layer_normalize=dec_layer_normalize,
        )
        hyperparams = model.get_hyperparameters()
        # Append to dictionary those hyperparams not stored internally.
        hyperparams.update({
            'phi_hidden_dims' : phi_hidden_dims,
            'phi_hidden_acts' : phi_hidden_acts,
            'phi_final_act' : phi_final_act,
            'phi_layer_normalize' : phi_layer_normalize,
            'enc_hidden_dims' : enc_hidden_dims,
            'enc_hidden_acts' : enc_hidden_acts,
            'enc_layer_normalize' : enc_layer_normalize,
            'dec_hidden_dims' : dec_hidden_dims,
            'dec_hidden_acts' : dec_hidden_acts,
            'dec_final_act' : dec_final_act,
            'dec_layer_normalize' : dec_layer_normalize,
        })
        model = jtu.tree_map(
            lambda x: x.astype(dtype) if eqx.is_array(x) else x, model
        )
        return model, hyperparams

    #############################
    ##  Initialization Method  ##
    #############################

    def initialize(
            self, 
            key, dtype=jnp.float32, *,
            init_enc_mlp_weights_method='xavier_uniform',
            init_enc_mlp_weights_args=[],
            init_enc_mlp_bias_method='constant',
            init_enc_mlp_bias_args=[0.],
            init_enc_mu_weights_method='xavier_uniform',
            init_enc_mu_weights_args=[],
            init_enc_mu_bias_method='constant',
            init_enc_mu_bias_args=[0.],
            init_enc_logvar_weights_method='xavier_uniform',
            init_enc_logvar_weights_args=[],
            init_enc_logvar_bias_method='constant',
            init_enc_logvar_bias_args=[0.],
            init_dec_weights_method='xavier_uniform',
            init_dec_weights_args=[],
            init_dec_bias_method='constant',
            init_dec_bias_args=[0.],
            **kwargs
    ) -> 'VAEPLNN':
        key, subkey = jrandom.split(key, 2)
        model = super().initialize(
            subkey, dtype=dtype, **kwargs
        )
        # Initialize encoder module
        model = self._initialize_linear_module(
            key, dtype, model, 'encoder_mlp', 
            init_weights_method=init_enc_mlp_weights_method,
            init_weights_args=init_enc_mlp_weights_args,
            init_biases_method=init_enc_mlp_bias_method,
            init_biases_args=init_enc_mlp_bias_args,
            include_biases=self.include_enc_bias,
        )
        model = self._initialize_linear_module(
            key, dtype, model, 'encoder_mu', 
            init_weights_method=init_enc_mu_weights_method,
            init_weights_args=init_enc_mu_weights_args,
            init_biases_method=init_enc_mu_bias_method,
            init_biases_args=init_enc_mu_bias_args,
            include_biases=self.include_enc_bias,
        )
        model = self._initialize_linear_module(
            key, dtype, model, 'encoder_logvar', 
            init_weights_method=init_enc_logvar_weights_method,
            init_weights_args=init_enc_logvar_weights_args,
            init_biases_method=init_enc_logvar_bias_method,
            init_biases_args=init_enc_logvar_bias_args,
            include_biases=self.include_enc_bias,
        )
        # Initialize decoder module
        model = self._initialize_linear_module(
            key, dtype, model, 'decoder', 
            init_weights_method=init_dec_weights_method,
            init_weights_args=init_dec_weights_args,
            init_biases_method=init_dec_bias_method,
            init_biases_args=init_dec_bias_args,
            include_biases=self.include_dec_bias,
        )
        return model
    
    ############################
    ##  Model Saving/Loading  ##
    ############################

    @staticmethod
    def load(fname:str, dtype=jnp.float32) -> tuple['VAEPLNN', dict]:
        """Load a model from a binary parameter file.
        
        Args:
            fname (str): Parameter file to load.
            dtype : Datatype. Either jnp.float32 or jnp.float64.

        Returns:
            VAEPLNN: Model instance.
            dict: Model hyperparameters.
        """
        with open(fname, "rb") as f:
            hyperparams = json.loads(f.readline().decode())
            model, _ = VAEPLNN.make_model(
                key=jrandom.PRNGKey(0), dtype=dtype, **hyperparams
            )
            return eqx.tree_deserialise_leaves(f, model), hyperparams
    
    ###################################
    ##  Construction Helper Methods  ##
    ###################################

    def _construct_encoder_module(
            self, key, hidden_dims, hidden_acts, 
            layer_normalize, bias=False, dtype=jnp.float32
    ):
        key, subkey1, subkey2, subkey3 = jrandom.split(key, 4)
        hidden_acts0 = None if hidden_acts is None else hidden_acts[:-1]
        hidden_acts1 = None if hidden_acts is None else hidden_acts[-1]
        encoder_mlp = self._construct_ffn(
            subkey1, self.ndims, 
            hidden_dims[-1], hidden_dims[:-1], hidden_acts0, 
            hidden_acts1, layer_normalize, bias, dtype,
        )
        encoder_mu = eqx.nn.Linear(
            hidden_dims[-1], self.latent_dim, 
            use_bias=bias, dtype=dtype, key=subkey2,
        )
        encoder_logvar = eqx.nn.Linear(
            hidden_dims[-1], self.latent_dim, 
            use_bias=bias, dtype=dtype, key=subkey3,
        )
        return encoder_mlp, encoder_mu, encoder_logvar
    
    def _construct_decoder_module(
            self, key, hidden_dims, hidden_acts, final_act, 
            layer_normalize, bias=False, dtype=jnp.float32
    ):
        key, subkey = jrandom.split(key, 2)
        decoder_mlp = self._construct_ffn(
            subkey, self.latent_dim, self.ndims, hidden_dims, hidden_acts, 
            final_act, layer_normalize, bias, dtype,
        )
        return decoder_mlp
