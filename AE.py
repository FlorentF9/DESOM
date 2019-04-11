"""
Implementation of the Deep Embedded Self-Organizing Map model
Autoencoder helper function

@author Florent Forest
@version 1.0
"""

from tensorflow import keras # using Tensorflow's Keras API
from keras.models import Model
from keras.layers import Input, Dense
import numpy as np

def mlp_autoencoder(encoder_dims, act='relu', init='glorot_uniform'):
    """
    Fully connected symmetric autoencoder model.

    # Arguments
        encoder_dims: list of number of units in each layer of encoder. encoder_dims[0] is input dim, encoder_dims[-1] is units in hidden layer (latent dim).
        The decoder is symmetric with encoder, so number of layers of the AE is 2*len(encoder_dims)-1
        act: activation of AE intermediate layers, not applied to Input, Hidden and Output layers
        init: initialization of AE layers
    # Return
        (ae_model, encoder_model, decoder_model): AE, encoder and decoder models
    """
    n_stacks = len(encoder_dims) - 1

    # Input
    x = Input(shape=(encoder_dims[0],), name='input')
    # Internal layers in encoder
    encoded = x
    for i in range(n_stacks-1):
        encoded = Dense(encoder_dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(encoded)
    # Hidden layer (latent space)
    encoded = Dense(encoder_dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(encoded) # hidden layer, latent representation is extracted from here
    # Internal layers in decoder
    decoded = encoded
    for i in range(n_stacks-1, 0, -1):
        decoded = Dense(encoder_dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(decoded)
    # Output
    decoded = Dense(encoder_dims[0], kernel_initializer=init, name='decoder_0')(decoded)

    # AE model
    autoencoder = Model(inputs=x, outputs=decoded, name='AE')

    # Encoder model
    encoder = Model(inputs=x, outputs=encoded, name='encoder')

    # Create input for decoder model
    encoded_input = Input(shape=(encoder_dims[-1],))
    # Internal layers in decoder
    decoded = encoded_input
    for i in range(n_stacks-1, -1, -1):
        decoded = autoencoder.get_layer('decoder_%d' % i)(decoded)
    # Decoder model
    decoder = Model(inputs=encoded_input, outputs=decoded, name='decoder')

    return (autoencoder, encoder, decoder)
