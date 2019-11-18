"""
Implementation of the Deep Embedded Self-Organizing Map model
Autoencoder helper functions

@author Florent Forest
@version 2.0
"""

from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
import numpy as np


def mlp_autoencoder(encoder_dims,
                    act='relu',
                    init='glorot_uniform',
                    dropout=1.0,
                    batchnorm=False):
    """Fully connected symmetric autoencoder model.

    Parameters
    ----------
    encoder_dims : list
        number of units in each layer of encoder. encoder_dims[0] is the input dim, encoder_dims[-1] is the
        size of the hidden layer (latent dim). The autoencoder is symmetric, so the total number of layers
        is 2*len(encoder_dims) - 1
    act : str (default='relu')
        activation of AE intermediate layers, not applied to Input, Hidden and Output layers
    init : str (default='glorot_uniform')
        initialization of AE layers
    dropout : float in [0, 1] (default=1.0)
        dropout keep probability
    batchnorm : bool (default=False)
        use batch normalization

    Returns
    -------
    ae_model, encoder_model, decoder_model : tuple
        autoencoder, encoder and decoder models
    """
    n_stacks = len(encoder_dims) - 1

    # Input
    x = Input(shape=(encoder_dims[0],), name='input')
    # Internal layers in encoder
    encoded = x
    for i in range(n_stacks-1):
        encoded = Dense(encoder_dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(encoded)
    # Hidden layer (latent space)
    encoded = Dense(encoder_dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(encoded)  # latent representation is extracted from here
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

    return autoencoder, encoder, decoder


def conv2d_autoencoder(input_shape,
                       latent_dim,
                       encoder_filters,
                       filter_size,
                       pooling_size,
                       act='relu',
                       dropout=1.0,
                       batchnorm=False):
    """2D convolutional autoencoder model.

    Parameters
    ----------
    input_shape : tuple
        input shape given as (height, width, channels) tuple
    latent_dim : int
        dimension of latent code (units in hidden dense layer)
    encoder_filters : list
        number of filters in each layer of encoder. The autoencoder is symmetric,
        so the total number of layers is 2*len(encoder_filters) - 1
    filter_size : int
        size of conv filters
    pooling_size : int
        size of maxpool filters
    act : str (default='relu')
        activation of AE intermediate layers, not applied to Input, Hidden and Output layers
    dropout : float in [0, 1] (default=1.0)
        dropout keep probability
    batchnorm : boolean (default=False)
        use batch normalization

    Returns
    -------
        ae_model, encoder_model, decoder_model : tuple
            autoencoder, encoder and decoder models
    """
    n_stacks = len(encoder_filters)

    # Infer code shape (assuming "same" padding, conv stride equal to 1 and max pooling stride equal to pool_size)
    code_shape = list(input_shape)
    for _ in range(n_stacks):
        code_shape[0] = int(np.ceil(code_shape[0] / pooling_size))
        code_shape[1] = int(np.ceil(code_shape[1] / pooling_size))
    code_shape[2] = encoder_filters[-1]

    # Input
    x = Input(shape=input_shape, name='input')
    # Internal layers in encoder
    encoded = x
    for i in range(n_stacks):
        encoded = Conv2D(encoder_filters[i], filter_size, activation=act, padding='same', name='encoder_conv_%d' % i)(encoded)
        encoded = MaxPooling2D(pooling_size, padding='same', name='encoder_maxpool_%d' % i)(encoded)
    # Flatten
    flattened = Flatten(name='flatten')(encoded)
    # Project using dense layer
    code = Dense(latent_dim, name='code')(flattened)  # latent representation is extracted from here
    # Reshape
    reshaped = Reshape(code_shape, name='reshape')(code)
    # Internal layers in decoder
    decoded = reshaped
    for i in range(n_stacks-1, -1, -1):
        decoded = Conv2D(encoder_filters[i], filter_size, activation=act, padding='same', name='decoder_conv_%d' % i)(
            decoded)
        # if i > 0:
        #     decoded = Conv2D(encoder_filters[i], filter_size, activation=act, padding='same', name='decoder_conv_%d' % i)(decoded)
        # else:
        #     decoded = Conv2D(encoder_filters[i], filter_size, activation=act, name='decoder_conv_%d' % i)(decoded)  # TODO CHECK
        decoded = UpSampling2D(pooling_size, name='decoder_upsample_%d' % i)(decoded)
    # Output
    decoded = Conv2D(1, filter_size, activation=act, padding='same', name='decoder_0')(decoded)

    # AE model
    autoencoder = Model(inputs=x, outputs=decoded, name='AE')

    # Encoder model (flattened output)
    encoder = Model(inputs=x, outputs=code, name='encoder')

    # Decoder model
    flattened_encoded_input = Input(shape=(latent_dim,))
    encoded_input = autoencoder.get_layer('reshape')(flattened_encoded_input)
    decoded = encoded_input
    for i in range(n_stacks-1, -1, -1):
        decoded = autoencoder.get_layer('decoder_conv_%d' % i)(decoded)
        decoded = autoencoder.get_layer('decoder_upsample_%d' % i)(decoded)
    decoded = autoencoder.get_layer('decoder_0')(decoded)
    decoder = Model(inputs=flattened_encoded_input, outputs=decoded, name='decoder')

    return autoencoder, encoder, decoder
