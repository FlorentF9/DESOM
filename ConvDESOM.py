"""
Implementation of the Convolutional Deep Embedded Self-Organizing Map model
Model file

@author Florent Forest
@version 2.0
"""

# Tensorflow/Keras
from keras.models import Model

# DESOM components
from SOM import SOMLayer
from AE import conv2d_autoencoder
from DESOM import DESOM


class ConvDESOM(DESOM):
    """Convolutional Deep Embedded Self-Organizing Map (ConvDESOM) model

    Example
    -------
    ```
    desom = desom = ConvDESOM(input_shape=X_train.shape[1:],
                              encoder_filters=[32, 64, 128, 256],
                              filter_size=3,
                              pooling_size=1,
                              map_size=(10, 10))
    ```

    Parameters
    ----------
    input_shape : tuple
        input shape given as (height, width) tuple
    latent_dim : int
        dimension of latent code (units in hidden dense layer)
    encoder_filters : list
        number of filters in each layer of encoder. The autoencoder is symmetric,
        so the total number of layers is 2*len(encoder_filters) - 1
    filter_size : int
        size of conv filters
    pooling_size : int
        size of maxpool filters
    map_size : tuple
        size of the rectangular map. Number of prototypes is map_size[0] * map_size[1]
    """

    def __init__(self, input_shape, latent_dim, encoder_filters, filter_size, pooling_size, map_size):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.encoder_filters = encoder_filters
        self.filter_size = filter_size
        self.pooling_size = pooling_size
        self.map_size = map_size
        self.n_prototypes = map_size[0] * map_size[1]
        self.pretrained = False
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.model = None

    def initialize(self, ae_act='relu', ae_init='glorot_uniform'):
        """Initialize ConvDESOM model

        Parameters
        ----------
        ae_act : str (default='relu')
            activation for AE intermediate layers
        ae_init : str (default='glorot_uniform')
            initialization of AE layers
        """
        # Create AE models
        self.autoencoder, self.encoder, self.decoder = conv2d_autoencoder(self.input_shape,
                                                                          self.latent_dim,
                                                                          self.encoder_filters,
                                                                          self.filter_size,
                                                                          self.pooling_size,
                                                                          ae_act,
                                                                          ae_init)
        som_layer = SOMLayer(self.map_size, name='SOM')(self.encoder.output)
        # Create ConvDESOM model
        self.model = Model(inputs=self.autoencoder.input,
                           outputs=[self.autoencoder.output, som_layer])
