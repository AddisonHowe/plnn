"""Autoencoder PLNN

"""

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

class AEPLNN(DeepPhiPLNN):

    encoder: eqx.Module
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
            enc_final_act,
            enc_layer_normalize,
            include_dec_bias,
            dec_hidden_dims,
            dec_hidden_acts,
            dec_final_act,
            dec_layer_normalize,
            **kwargs
    ):
        key, subkey = jrandom.split(key, 2)
        super().__init__(subkey, dtype=dtype, **kwargs)
        
        self.latent_dim = latent_dim
        self.include_enc_bias = include_enc_bias
        self.include_dec_bias = include_dec_bias

        key, key_enc, key_dec = jax.random.split(key, 3)

        # Encoder Neural Network: Maps ndims to latent_dim
        self.encoder = self._construct_encoder_module(
            key_enc, 
            enc_hidden_dims, 
            enc_hidden_acts, 
            enc_final_act, 
            enc_layer_normalize,
            bias=include_enc_bias, 
            dtype=dtype
        )

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
        batch_keys = jax.random.split(key, t0.shape[0])
        
        # Encode the inputs
        z0 = encvec(y0)
        # Evolve in latent space
        z1 = fwdvec(t0, t1, z0, sigparams, batch_keys)
        # Decode the evolved inputs
        y1 = decvec(z1)
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
        enc_linlayers = self._get_linear_module_layers(self.encoder)
        dec_linlayers = self._get_linear_module_layers(self.decoder)
        d.update({
            'encoder.w' : [l.weight for l in enc_linlayers],
            'encoder.b' : [l.bias for l in enc_linlayers],
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
        enc_linlayers  = self._get_linear_module_layers(self.encoder)
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

    def encode(self, y):
        z = self.encoder(y)
        return z
    
    def decode(self, z):
        return self.decoder(z)
    
    def encode_ensemble(self, ys):
        zs = jax.vmap(self.encoder, 0)(ys)
        return zs
    
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
        enc_final_act=None,
        enc_layer_normalize=False,
        include_dec_bias=True,
        dec_hidden_dims=[],
        dec_hidden_acts=None,
        dec_final_act=None,
        dec_layer_normalize=False,
    ) -> tuple['AEPLNN', dict]:
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
            enc_final_act
            enc_layer_normalize
            include_dec_bias
            dec_hidden_dims
            dec_hidden_acts
            dec_final_act
            dec_layer_normalize
        
        Returns:
            AEPLNN: Model instance.
            dict: Dictionary of hyperparameters.
        """
        model = AEPLNN(
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
            enc_final_act=enc_final_act,
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
            'enc_hidden_dims' : enc_hidden_dims,
            'enc_hidden_acts' : enc_hidden_acts,
            'enc_final_act' : enc_final_act,
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
            init_enc_weights_method='xavier_uniform',
            init_enc_weights_args=[],
            init_enc_bias_method='constant',
            init_enc_bias_args=[0.],
            init_dec_weights_method='xavier_uniform',
            init_dec_weights_args=[],
            init_dec_bias_method='constant',
            init_dec_bias_args=[0.],
            **kwargs
    ) -> 'AEPLNN':
        key, subkey = jrandom.split(key, 2)
        model = super().initialize(
            subkey, dtype=dtype, **kwargs
        )
        # Initialize encoder module
        model = self._initialize_linear_module(
            key, dtype, model, 'encoder', 
            init_weights_method=init_enc_weights_method,
            init_weights_args=init_enc_weights_args,
            init_biases_method=init_enc_bias_method,
            init_biases_args=init_enc_bias_args,
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
    
    ###################################
    ##  Construction Helper Methods  ##
    ###################################

    def _construct_encoder_module(
            self, key, hidden_dims, hidden_acts, final_act, 
            layer_normalize, bias=False, dtype=jnp.float32
    ):
        key, subkey = jrandom.split(key, 2)
        encoder_mlp = self._construct_ffn(
            subkey, self.ndims, self.latent_dim, hidden_dims, hidden_acts, 
            final_act, layer_normalize, bias, dtype,
        )
        return encoder_mlp
    
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
